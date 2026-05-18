#!/bin/bash

set -e
set -x

python -m pip install cibuildwheel
python -m cibuildwheel build/python --output-dir wheelhouse
