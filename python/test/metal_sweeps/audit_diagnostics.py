"""Which named, fail-closed diagnostics does a test actually pin?

Counting `notifyMatchFailure` says nothing: most of the ~275 in the conversion
pass mean "this pattern does not apply, try the next one" and are invisible to
a user. What is measurable is the NAMED diagnostics -- the ones that end
compilation with a reason -- and whether a test holds each of them in place.

Two traps this encodes, both of which cost a round to learn:

1. Not every `reject(...)` emits. `walkBackThroughElementwiseChain` has a local
   lambda of that name which only writes a `reason` string, and since the
   rank-1 reduce message is built from the cone predicate's failure point
   instead, those strings reach no user at all. Counting them inflates the
   total by a third.

2. **Unpinned is not reachable.** A diagnostic can be unreachable because the
   Triton frontend refuses the shape first, or because an earlier and more
   specific diagnostic shadows it. Pinning those fills the suite with shapes a
   user cannot write. Probe before pinning -- see `atomic` in the notes below.

Usage:
    pixi run --frozen python python/test/metal_sweeps/audit_diagnostics.py
    pixi run --frozen python python/test/metal_sweeps/audit_diagnostics.py --list reduce
"""

import ast
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
SRC = os.path.join(
    REPO, "third_party/metal/lib/Conversion/TritonGPUToMetal/TritonGPUToMetal.cpp")


def _emitting_reject_spans(text):
    """Byte ranges of `reject` lambdas that actually emit a diagnostic."""
    spans = []
    for m in re.finditer(r"auto reject = \[&\]\((.*?)\)\s*\{", text, re.S):
        j, depth = m.end(), 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        body = text[m.end():j]
        spans.append((m.start(), "emitOpError" in body or "emitError" in body))
    return spans


def collect(text):
    """Named diagnostics -> the lines they are emitted from."""
    spans = _emitting_reject_spans(text)

    def reject_emits(pos):
        recent = [(start, emits) for start, emits in spans if start < pos]
        return bool(recent) and recent[-1][1]

    lit = re.compile(r'"((?:[^"\\]|\\.)*)"')
    diags = {}
    for m in re.finditer(r"(emitOpError|emitError|reject)\s*\(", text, re.S):
        if m.group(1) == "reject" and not reject_emits(m.start()):
            continue
        i, depth, j = m.end(), 1, m.end()
        while j < len(text) and depth:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        parts = [p.group(1) for p in lit.finditer(text[i:j])]
        if not parts:
            continue
        msg = re.sub(r"\s+", " ",
                     "".join(parts).replace("\\n", " ").replace('\\"', '"')).strip()
        if len(msg) >= 20:
            diags.setdefault(msg, []).append(text.count("\n", 0, m.start()) + 1)
    return diags


def _words(text):
    """Words with punctuation dropped, not merely split on whitespace.

    ⚠️ Splitting on whitespace alone reports a pinned diagnostic as unpinned.
    `test_atomic_add_bf16_is_rejected_with_a_reason` quotes exactly the five
    words "bf16 atomic add is unsupported", and in the message the fifth is
    "unsupported:" -- so every five-word window carried a colon the test does
    not have. The first round of this audit called that diagnostic unreachable
    and spent a probe looking for a shape to trigger a test that existed.
    """
    return re.sub(r"[^0-9A-Za-z]+", " ", text).split()


def haystack():
    """Only what a test could ASSERT on: string literals and lit expectations.

    ⚠️ Reading whole files counts a test's own COMMENTS as coverage. Ordinary
    English is enough: "how many elements a thread owns" appears in a comment
    about register bands, and a five-word run of it marked a diagnostic no test
    mentions as pinned. A pin is an assertion, so only Python string constants
    and lit match payloads count here.
    """
    out = []
    for pat in ("python/test/unit/**/*.py", "python/test/regression/**/*.py"):
        for f in glob.glob(os.path.join(REPO, pat), recursive=True):
            try:
                tree = ast.parse(open(f, errors="ignore").read())
            except (OSError, SyntaxError):
                continue
            # ⚠️ Docstrings are string constants too, and a test's docstring
            # explains the shape in the same words as the diagnostic. Only a
            # string a test can COMPARE against counts.
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                    body = getattr(node, "body", None)
                    if (body and isinstance(body[0], ast.Expr) and
                            isinstance(body[0].value, ast.Constant) and
                            isinstance(body[0].value.value, str)):
                        docstrings.add(id(body[0].value))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant) and
                        isinstance(node.value, str) and
                        id(node) not in docstrings):
                    out.append(" ".join(_words(node.value)))
    for f in glob.glob(os.path.join(REPO, "test/**/*.mlir"), recursive=True):
        try:
            text = open(f, errors="ignore").read()
        except OSError:
            continue
        for m in re.finditer(r"\{\{(.*?)\}\}", text, re.S):
            out.append(" ".join(_words(m.group(1))))
        for m in re.finditer(r"^\s*//\s*CHECK:\s*(.*)$", text, re.M):
            out.append(" ".join(_words(m.group(1))))
    return "\n".join(out)


def pinned(msg, hay):
    """A test pins a diagnostic if it quotes a long enough run of its words.

    Whole-string matching does not work: several diagnostics interpolate a
    runtime value, and tests quote the fixed part. Empty quotes in the collected
    message mark a streamed value, such as an extern symbol; its surrounding
    diagnostic may also stream arities. Match through those values without
    crossing from one assertion into another.
    """
    if "''" in msg:
        pattern = "[0-9A-Za-z ]{0,48}?".join(
            re.escape(word) for word in _words(msg))
        if re.search(pattern, hay):
            return True
    words = _words(msg)
    for n in (8, 6, 5):
        for k in range(len(words) - n + 1):
            if " ".join(words[k:k + n]) in hay:
                return True
    return False


AREAS = (
    ("dot_scaled", ("scaled",)),
    ("exchange/staging", ("exchange", "staging", "threadgroup")),
    ("dot/matmul", ("dot", "matmul")),
    ("reduce/scan", ("reduce", "scan")),
    ("atomic", ("atomic",)),
    ("layout/tile", ("layout", "tile", "blocked")),
)


def area(msg):
    for name, keys in AREAS:
        if any(k in msg for k in keys):
            return name
    return "other"


def main():
    text = open(SRC).read()
    diags = collect(text)
    hay = haystack()
    unpinned = {m: l for m, l in diags.items() if not pinned(m, hay)}
    want = None
    if "--list" in sys.argv:
        want = sys.argv[sys.argv.index("--list") + 1]
    print(f"named fail-closed diagnostics: {len(diags)}")
    print(f"  pinned by a test: {len(diags) - len(unpinned)}")
    print(f"  NOT pinned:       {len(unpinned)}")
    counts = {}
    for msg in unpinned:
        counts.setdefault(area(msg), []).append(msg)
    print("\nunpinned by area:")
    for name, msgs in sorted(counts.items(), key=lambda kv: -len(kv[1])):
        print(f"  {name:18} {len(msgs)}")
    if want:
        print(f"\n--- unpinned matching {want!r} ---")
        for msg, lines in sorted(unpinned.items()):
            if want in msg or want in area(msg):
                print(f"  L{lines[0]:<6} {msg[:100]}")
    out = os.path.join(REPO, "build", "metal-unpinned-diagnostics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({m: l for m, l in sorted(unpinned.items())}, open(out, "w"), indent=1)
    print(f"\nfull list: {out}")


if __name__ == "__main__":
    main()
