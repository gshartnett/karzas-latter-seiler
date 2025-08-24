import os
import sys

import toml

# Add your project root to Python path
sys.path.insert(0, os.path.abspath("../../"))

# Read version from pyproject.toml
with open(os.path.abspath("../../pyproject.toml"), "r") as f:
    pyproject = toml.load(f)

project_version = pyproject["tool"]["poetry"]["version"]
project_name = pyproject["tool"]["poetry"]["name"]
project_authors = pyproject["tool"]["poetry"]["authors"]

# -- Project information -----------------------------------------------------
project = "Karzas-Latter-Seiler EMP Model"
copyright = "2024, Gavin S. Hartnett"
author = "Gavin S. Hartnett"
release = project_version
version = project_version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Auto-generate API docs
    "sphinx.ext.autosummary",  # Generate summary tables
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.napoleon",  # Support for Google/NumPy docstrings
    "sphinx.ext.mathjax",  # Math support
    "sphinx.ext.intersphinx",  # Link to other docs
    "sphinx_autodoc_typehints",  # Better type hints
]

# Add MyST parser for Markdown support (optional)
# extensions.append('myst_parser')

# Napoleon settings for docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Autosummary settings
autosummary_generate = True

# Add any paths that contain templates here
templates_path = ["_templates"]

# List of patterns to ignore when looking for source files
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# -- Extension configuration -------------------------------------------------
# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
