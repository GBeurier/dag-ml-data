from __future__ import annotations

project = "dag-ml-data"
author = "G. Beurier"
copyright = "2026, G. Beurier"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "design/source/*",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 3

html_theme = "alabaster"
html_title = "dag-ml-data"
html_static_path: list[str] = []

# The first public docs navigation is intentionally smaller than the complete
# historical design archive. The build still validates linked pages.
suppress_warnings = ["toc.not_included"]
