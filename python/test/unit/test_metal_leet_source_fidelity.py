from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "python" / "test" / "unit" / "fixtures" / "metal_leet"
SOURCE = ROOT / "leetgpu-triton-solve"
MANIFEST = FIXTURES / "source_fidelity.json"
INVENTORY = ROOT / "python" / "test" / "unit" / "test_metal_backend_leet_uncovered.py"

SOURCE_CLASSES = {
    "verbatim_kernel",
    "host_only_adaptation",
    "source_repair",
    "metal_specific_kernel_rewrite",
}


def _manifest():
    return json.loads(MANIFEST.read_text())


def _fixture_files():
    return {
        path.relative_to(FIXTURES).as_posix()
        for path in FIXTURES.rglob("*.py")
    }


def _source_path(manifest, relative):
    return manifest.get("source_name_overrides", {}).get(relative, relative)


def _literal_assignment(tree, name):
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} assignment not found in {INVENTORY}")


def _owned_fixtures():
    tree = ast.parse(INVENTORY.read_text())
    standalone = set(_literal_assignment(tree, "STANDALONE_FIXTURES"))
    targeted = set(_literal_assignment(tree, "TARGETED_FIXTURES"))
    interpreter = {
        f"{name}.py" for name in _literal_assignment(tree, "UNRUNNABLE")
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "_case"
                and len(decorator.args) == 1
            ):
                interpreter.add(f"{ast.literal_eval(decorator.args[0])}.py")

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "CASES"
        ):
            interpreter.add(f"{ast.literal_eval(node.targets[0].slice)}.py")

    categories = (standalone, targeted, interpreter)
    for index, category in enumerate(categories):
        others = set().union(
            *(other for i, other in enumerate(categories) if i != index)
        )
        assert category.isdisjoint(others)
    return set().union(*categories)


def _jit_functions(path: Path):
    tree = ast.parse(path.read_text())
    functions = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if any(
            "triton" in ast.dump(deco, include_attributes=False)
            and "jit" in ast.dump(deco, include_attributes=False)
            for deco in node.decorator_list
        ):
            functions.append(ast.dump(node, include_attributes=False))
    return functions


def test_source_fidelity_manifest_is_complete():
    manifest = _manifest()
    classes = manifest["classifications"]
    assert manifest["schema_version"] == 1
    assert set(classes) == SOURCE_CLASSES
    assert manifest["source_commit"] == (
        "4dc4e6931f5f534981b327522464b3bebcefead1"
    )
    assert manifest["source_top_level_python_files"] == 88
    assert manifest["excluded_sources"] == [
        {"path": "easy-matrx_copy.py", "reason": "zero-byte source file"}
    ]

    seen = []
    for relative_list in [*classes.values(), manifest["fixture_only"]]:
        assert relative_list
        seen.extend(relative_list)
    assert len(seen) == len(set(seen))
    assert set(seen) == _fixture_files()
    assert set(seen) == _owned_fixtures()

    leet_mapped = set().union(*(set(classes[name]) for name in SOURCE_CLASSES))
    fixture_only = set(manifest["fixture_only"])
    assert len(leet_mapped) == 87
    assert len(fixture_only) == 3
    assert leet_mapped.isdisjoint(fixture_only)


def test_source_fidelity_matches_pinned_checkout_when_available():
    manifest = _manifest()
    if not SOURCE.is_dir():
        pytest.skip("leetgpu-triton-solve checkout is not present")
    head = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != manifest["source_commit"]:
        pytest.skip(f"source checkout is {head}, not pinned manifest commit")

    assert (SOURCE / "easy-matrx_copy.py").is_file()
    assert (SOURCE / "easy-matrx_copy.py").stat().st_size == 0

    actual_sources = {
        path.name for path in SOURCE.glob("*.py") if path.stat().st_size > 0
    }
    mapped_sources = {
        _source_path(manifest, relative)
        for class_name in SOURCE_CLASSES
        for relative in manifest["classifications"][class_name]
    }
    assert mapped_sources == actual_sources

    for relative in manifest["fixture_only"]:
        assert not (SOURCE / relative).exists()

    for relative in manifest["classifications"]["verbatim_kernel"]:
        source = SOURCE / _source_path(manifest, relative)
        assert (FIXTURES / relative).read_bytes() == source.read_bytes(), relative

    for relative in manifest["classifications"]["host_only_adaptation"]:
        source = SOURCE / _source_path(manifest, relative)
        assert (FIXTURES / relative).read_bytes() != source.read_bytes(), relative
        assert _jit_functions(FIXTURES / relative) == _jit_functions(source), relative

    for class_name in ("source_repair", "metal_specific_kernel_rewrite"):
        for relative in manifest["classifications"][class_name]:
            source = SOURCE / _source_path(manifest, relative)
            try:
                source_jit = _jit_functions(source)
            except SyntaxError:
                continue
            assert _jit_functions(FIXTURES / relative) != source_jit, relative
