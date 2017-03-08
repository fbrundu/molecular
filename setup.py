#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
  readme = readme_file.read()

with open('HISTORY.rst') as history_file:
  history = history_file.read()

requirements = [
  'bioservices==1.4.16',
  'pandas==0.19.2',
  'scipy==0.18.1',
  'plotly==2.0.5',
  'seaborn==0.7.1',
  'IPython==5.3.0',
  'statsmodels==0.8.0',
  'numpy==1.12.0',
  'ipywidgets==6.0.0',
]

test_requirements = [
  # TODO: put package test requirements here
]

setup(
  name='molecular',
  version='0.1.0',
  description="Computational Biology utility module",
  long_description=readme + '\n\n' + history,
  author="Francesco G. Brundu",
  author_email='francesco.brundu@gmail.com',
  url='https://github.com/fbrundu/molecular',
  packages=[
    'molecular',
    'molecular.histology',
    'molecular.plotting',
    'molecular.preprocessing',
  ],
#  package_dir={'molecular':
#               'molecular'},
  include_package_data=True,
  install_requires=requirements,
  license="MIT license",
  zip_safe=False,
  keywords='molecular',
  classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Programming Language :: Python :: 3.6',
  ],
  test_suite='tests',
  tests_require=test_requirements
)
