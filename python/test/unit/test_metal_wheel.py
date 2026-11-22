from __future__ import annotations

import os
import platform
import re
import subprocess
import sysconfig
import zipfile
from pathlib import Path

import pytest

from build_helpers import (
    configure_macos_build_architecture,
    resolve_macos_deployment_target,
    validate_metal_wheel_platform_tag,
)

MACHO_MAGICS = {
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
}


def test_metal_only_wheel_defaults_to_supported_macos_floor():
    assert resolve_macos_deployment_target(["metal"], system="Darwin") == "15.0"


def test_metal_only_wheel_preserves_newer_requested_target():
    assert resolve_macos_deployment_target(["metal"], requested_target="16.1", system="Darwin") == "16.1"


def test_metal_only_wheel_rejects_target_below_dependency_floor():
    with pytest.raises(RuntimeError, match="macOS 15.0 or newer"):
        resolve_macos_deployment_target(["metal"], requested_target="14.6", system="Darwin")


@pytest.mark.parametrize(
    "wheel_backends,system,requested_target,expected",
    [
        (["metal"], "Linux", None, None),
        (["metal", "nvidia"], "Darwin", None, None),
        (["metal", "nvidia"], "Darwin", "13.0", "13.0"),
    ],
)
def test_deployment_target_policy_is_scoped_to_darwin_metal_only_wheels(
    wheel_backends, system, requested_target, expected
):
    assert (
        resolve_macos_deployment_target(
            wheel_backends,
            requested_target=requested_target,
            system=system,
        )
        == expected
    )


def test_metal_wheel_tag_must_match_built_binary_floor():
    validate_metal_wheel_platform_tag("macosx_15_0_arm64", "15.0")
    with pytest.raises(RuntimeError, match="macosx_15_0_arm64"):
        validate_metal_wheel_platform_tag("macosx_26_0_arm64", "15.0")
    with pytest.raises(RuntimeError, match="arm64"):
        validate_metal_wheel_platform_tag("macosx_15_0_x86_64", "15.0")
    with pytest.raises(RuntimeError, match="arm64"):
        validate_metal_wheel_platform_tag("macosx_15_0_universal2", "15.0")


@pytest.mark.skipif(platform.system() != "Darwin", reason="requires macOS platform tagging")
@pytest.mark.parametrize("deployment_target", ["15.0", "26.0"])
def test_metal_wheel_uses_arm64_tag_with_universal2_python(monkeypatch, tmp_path, deployment_target):
    from setuptools import Distribution, Extension
    try:
        from setuptools.command.bdist_wheel import bdist_wheel
    except ImportError:
        from wheel.bdist_wheel import bdist_wheel

    monkeypatch.delenv("_PYTHON_HOST_PLATFORM", raising=False)
    monkeypatch.delenv("ARCHFLAGS", raising=False)
    monkeypatch.setenv("MACOSX_DEPLOYMENT_TARGET", deployment_target)
    config = sysconfig.get_config_vars()
    monkeypatch.setitem(config, "MACOSX_DEPLOYMENT_TARGET", "10.13")
    monkeypatch.setitem(config, "CFLAGS", "-arch x86_64 -arch arm64")
    monkeypatch.setitem(config, "_OSX_SUPPORT_INITIAL_CFLAGS", "-arch x86_64 -arch arm64")
    assert sysconfig.get_platform() == "macosx-10.13-universal2"

    wheel = bdist_wheel(Distribution({"ext_modules": [Extension("test_extension", sources=[])]}))
    wheel.ensure_finalized()
    wheel.bdist_dir = str(tmp_path)
    assert wheel.get_tag()[2].endswith("_universal2")

    architecture = configure_macos_build_architecture(["metal"], deployment_target)
    assert architecture == "arm64"
    assert os.environ["ARCHFLAGS"] == "-arch arm64"
    validate_metal_wheel_platform_tag(wheel.get_tag()[2], deployment_target)


@pytest.mark.parametrize("wheel_backends,system", [(["metal"], "Linux"), (["metal", "nvidia"], "Darwin")])
def test_macos_architecture_policy_is_scoped_to_metal_wheels(monkeypatch, wheel_backends, system):
    monkeypatch.delenv("_PYTHON_HOST_PLATFORM", raising=False)
    monkeypatch.delenv("ARCHFLAGS", raising=False)
    assert configure_macos_build_architecture(wheel_backends, None, system=system) is None
    assert "_PYTHON_HOST_PLATFORM" not in os.environ
    assert "ARCHFLAGS" not in os.environ


def test_metal_wheel_preserves_explicit_architecture_overrides(monkeypatch):
    monkeypatch.setenv("_PYTHON_HOST_PLATFORM", "macosx-15.0-universal2")
    monkeypatch.setenv("ARCHFLAGS", "-arch arm64 -arch x86_64")
    configure_macos_build_architecture(["metal"], "15.0", system="Darwin")
    assert os.environ["_PYTHON_HOST_PLATFORM"] == "macosx-15.0-universal2"
    assert os.environ["ARCHFLAGS"] == "-arch arm64 -arch x86_64"


@pytest.fixture(scope="module")
def built_metal_wheel() -> Path:
    wheel = os.environ.get("TRITON_TEST_METAL_WHEEL")
    if wheel is None:
        pytest.skip("set TRITON_TEST_METAL_WHEEL to validate a built artifact")
    path = Path(wheel)
    assert path.is_file(), f"Metal wheel does not exist: {path}"
    return path


def test_built_metal_wheel_has_exact_platform_contract(built_metal_wheel):
    assert built_metal_wheel.name.endswith("-cp312-abi3-macosx_15_0_arm64.whl")
    with zipfile.ZipFile(built_metal_wheel) as archive:
        wheel_metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/WHEEL"))
        wheel_metadata = archive.read(wheel_metadata_name).decode()
    assert "Tag: cp312-abi3-macosx_15_0_arm64" in wheel_metadata


def test_built_metal_wheel_contains_only_metal_backend_payload(built_metal_wheel):
    with zipfile.ZipFile(built_metal_wheel) as archive:
        names = archive.namelist()
        entry_points_name = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        entry_points = archive.read(entry_points_name).decode()

    backend_packages = {match.group(1) for name in names if (match := re.match(r"triton/backends/([^/]+)/", name))}
    assert backend_packages == {"metal"}
    assert "metal = triton.backends.metal" in entry_points
    assert "nvidia = triton.backends.nvidia" not in entry_points
    assert "amd = triton.backends.amd" not in entry_points
    assert not any(name.startswith("triton/profiler/") for name in names)
    assert not any("ptxas" in name or "cupti" in name for name in names)


def test_every_built_metal_wheel_macho_file_targets_macos_15(built_metal_wheel, tmp_path):
    with zipfile.ZipFile(built_metal_wheel) as archive:
        macho_files = {
            name: contents
            for name in archive.namelist()
            if not name.endswith("/") and (contents := archive.read(name))[:4] in MACHO_MAGICS
        }
        assert "triton/_C/libtriton.so" in macho_files
        for index, (name, contents) in enumerate(macho_files.items()):
            output = tmp_path / f"{index}-{Path(name).name}"
            output.write_bytes(contents)
            architectures = subprocess.check_output(["lipo", "-archs", output], text=True).split()
            assert architectures == ["arm64"], f"{name}: {architectures}"
            load_commands = subprocess.check_output(["otool", "-l", output], text=True)
            min_versions = re.findall(r"\n\s+minos (\S+)", load_commands)
            assert min_versions == ["15.0"], f"{name}: {min_versions}"
