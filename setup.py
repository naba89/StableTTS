#!/usr/bin/env python

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="stable_tts",
    version="0.0.1",
    description="StableTTS pip installable",
    author="Nabarun Goswami",
    author_email="nabarungoswami@mi.t.u-tokyo.ac.jp",
    packages=find_packages(),
    package_data={
        'stable_tts': [
            'text/cnm3/*',  # Include all files in text/cnm3 directory
            'text/langdetect/profiles/*',  # Include all files in text/langdetect/profiles directory
            'text/langdetect/utils/messages.properties',  # Include messages.properties file in text/langdetect/utils directory
        ],
    },
    install_requires=required
)
