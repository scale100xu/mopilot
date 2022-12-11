#!/usr/bin/env bash
####
## This is Test version
## release version is https://pypi.org
####
rm -rf build
rm -rf dist
rm -rf mopilot.egg-info
# compile python package
python3 setup.py sdist bdist_wheel
# upload python package files, dist/*,you need your account
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose