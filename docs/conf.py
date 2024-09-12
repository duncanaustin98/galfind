# Configuration file for the Sphinx documentation builder.

# -- Project information

from galfind._version import __version__

project = 'galfind'
copyright = '2024, Austin'
author = 'Duncan Austin'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx-rtd-theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'