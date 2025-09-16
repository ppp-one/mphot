# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import importlib.metadata

# add path to source code
sys.path.insert(0, os.path.abspath("../../src/"))

project = "mphot"
copyright = "2025, Peter Pedersen"
author = "Peter Pedersen"
version = importlib.metadata.version("mphot")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "myst_nb",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_title = "mphot"
html_static_path = ["_static"]
html_context = {
    # ...
    "default_mode": "light"
}

html_theme_options = {
    "repository_url": "https://github.com/ppp-one/mphot",
    "use_repository_button": True,
    "use_fullscreen_button": False,
    "use_download_button": False,
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
    "navbar_end": ["navbar-icon-links"],
}

# Auto-generate API documentation
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": True,
}
autosummary_generate = True
autodoc_preserve_defaults = True  # Prevents evaluation of default values
autoclass_content = "both"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False
