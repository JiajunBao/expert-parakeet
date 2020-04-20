#!/usr/bin/env bash
set -x
/opt/anaconda/bin/python3 src/create_input_files.py
/opt/anaconda/bin/python3 src/train.py > log2.out
