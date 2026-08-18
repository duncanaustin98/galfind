# Configuration file for the Sphinx documentation builder.

# -- Project information

# from galfind._version import __version__

project = "galfind"
copyright = "2024, Duncan Austin"
author = "Duncan Austin"

release = "0.1"
version = "0.1.0"

# -- General configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    #"sphinx.ext.highlighting",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

numpydoc_show_class_members = False
class_members_toctree = False
nbsphinx_execute = "never"
autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

master_doc = "index"

# Autodoc configuration for better API documentation
autodoc_default_options = {
    'undoc-members': False,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# Configure Napoleon for NumPy-style docstrings
napoleon_include_init_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_keyword = True

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
