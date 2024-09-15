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
]

numpydoc_show_class_members = False
class_members_toctree = False
nbsphinx_allow_errors = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

master_doc = "index"

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
