# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'Marketing Mix Modeling avec PySpark'
copyright = '2025, Samir Elaissaouy'
author = 'Samir Elaissaouy'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Output file base name
htmlhelp_basename = 'MMM-PySpark-doc'

# Options for LaTeX output
latex_elements = {
    # Additional options for LaTeX
}

# Grouping the document tree into LaTeX files
latex_documents = [
    (master_doc, 'MMM-PySpark.tex', 'Marketing Mix Modeling avec PySpark Documentation',
     'Samir Elaissaouy', 'manual'),
]

# Options for manual page output
man_pages = [
    (master_doc, 'mmm-pyspark', 'Marketing Mix Modeling avec PySpark Documentation',
     [author], 1)
]

# Options for Texinfo output
texinfo_documents = [
    (master_doc, 'MMM-PySpark', 'Marketing Mix Modeling avec PySpark Documentation',
     author, 'MMM-PySpark', 'Marketing Mix Modeling avec PySpark.',
     'Miscellaneous'),
]

# Add any paths that contain templates
templates_path = ['_templates']

# Add any paths that contain custom static files
html_static_path = ['_static']
