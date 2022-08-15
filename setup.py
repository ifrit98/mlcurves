#!/usr/bin/python3
import os
from warnings import warn
from distutils.core import setup
from setuptools import find_packages, find_namespace_packages

REQ_FILE = 'requirements.txt'

if not os.path.exists(REQ_FILE):
      warn("No requirements file found.  Using defaults deps")
      deps = [
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "scikit-learn",
            "tensorflow<=2.8.*",
            "tensorflow_probability",
            "tensorflow_addons",
            "keras<=2.8"
      ]
      warn(', '.join(deps))
else:
      with open(REQ_FILE, 'r') as f:
            deps = f.read().splitlines()


setup(name='mlcurves',
      version='1.0.0',
      description='MLCurves: a lightweight module for producing training curves in ML development',
      author='Jason St George',
      author_email='stgeorge@brsc.com',
      packages=find_packages(),
      install_requires=deps
)
