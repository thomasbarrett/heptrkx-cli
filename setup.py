#!/usr/bin/env python

from setuptools import setup

setup(name="heptrkx-cli",
    version="1.0",
    description="Command Line Interface for HEP.trkx project",
    author="Thomas Barrett",
    author_email="tbarrett@caltech.edu",
    scripts=['bin/heptrkx-cli'],
    install_requires=[
        'trackml'
    ],
    dependency_links=[
        'https://github.com/LAL/trackml-library/tarball/master#egg=trackml-library',
    ]
)