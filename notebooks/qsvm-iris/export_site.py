"""Export the QSVM notebook as CSP-clean HTML for the portfolio site.

Converts ``qsvm_iris.ipynb`` with nbconvert's chrome-free ``basic`` template
(plots inlined as base64 ``data:`` images), then renders every LaTeX math span
to MathML at export time — browsers render MathML natively, so the page ships
zero JavaScript and satisfies the site's strict Content-Security-Policy
(no MathJax CDN, no inline styles).

The output lands in the website checkout (``make export-site``), where the
AI/ML umbrella page embeds it. A provenance comment at the top records the
source commit, date, and library versions, mirroring this repo's
``classifiers/web_export.py`` convention.
"""

from __future__ import annotations

import html
import importlib.metadata
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from latex2mathml.converter import convert as latex_to_mathml
from nbconvert import HTMLExporter

NB_DIR = Path(__file__).resolve().parent
NOTEBOOK = NB_DIR / "qsvm_iris.ipynb"
# The quantum-machine-learning repo root — provenance stamps its HEAD.
REPO_ROOT = NB_DIR.parents[1]
DEFAULT_OUT = REPO_ROOT.parent / "website" / "src" / "content" / "notebooks" / "qsvm-iris.html"

# Segments whose text must never be treated as math.
_PROTECTED = re.compile(r"<pre\b.*?</pre>|<code\b.*?</code>", re.DOTALL)
# Display math first (non-greedy), then inline; lone dollars are left alone.
_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")


def _mathml(latex_src: str, *, display: bool) -> str:
    """Render one LaTeX span to MathML (the HTML is entity-escaped; undo it)."""
    return latex_to_mathml(
        html.unescape(latex_src).strip(), display="block" if display else "inline"
    )


def _convert_math(segment: str) -> str:
    segment = _DISPLAY.sub(lambda m: _mathml(m.group(1), display=True), segment)
    return _INLINE.sub(lambda m: _mathml(m.group(1), display=False), segment)


def render_math(body: str) -> str:
    """Convert $...$ / $$...$$ to MathML everywhere outside pre/code blocks."""
    parts: list[str] = []
    last = 0
    for protected in _PROTECTED.finditer(body):
        parts.append(_convert_math(body[last : protected.start()]))
        parts.append(protected.group(0))
        last = protected.end()
    parts.append(_convert_math(body[last:]))
    return "".join(parts)


def _git(*args: str) -> str:
    result = subprocess.run(  # noqa: S603 — fixed argv, no user input
        ["git", *args],  # noqa: S607 — git from PATH is the intended toolchain
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _provenance() -> str:
    versions = ", ".join(
        f"{pkg} {importlib.metadata.version(pkg)}" for pkg in ("qiskit", "numpy", "nbconvert")
    )
    return (
        "<!-- Exported from quantum-machine-learning/notebooks/qsvm-iris/qsvm_iris.ipynb\n"
        "     by export_site.py\n"
        "     source_repo: quantum-machine-learning\n"
        f"     source_sha: {_git('rev-parse', 'HEAD')}\n"
        f"     source_dirty: {bool(_git('status', '--porcelain'))}\n"
        f"     exported_at: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}\n"
        f"     python {platform.python_version()}; {versions} -->\n"
    )


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    exporter = HTMLExporter(template_name="basic")
    body, _resources = exporter.from_filename(str(NOTEBOOK))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_provenance() + render_math(body))
    size_kb = out.stat().st_size // 1024
    sys.stdout.write(f"wrote {out} ({size_kb} KB)\n")


if __name__ == "__main__":
    main()
