#!/usr/bin/env python

from setuptools import setup

setup(name="heptrkx-cli",
    version="1.0",
    description="Command Line Interface for HEP.trkx project",
    author="Thomas Barrett",
    author_email="tbarrett@caltech.edu",
    scripts=['bin/heptrkx'],
    packages=['heptrkxcli'],
    install_requires=[
        'trackml',
        'pyyaml',
        'numpy',
        'pandas',
        'tensorflow-gpu',
        'graph-nets'
    ],
    dependency_links=[
        'https://github.com/LAL/trackml-library/tarball/master#egg=trackml-library',
    ]
)