#!/usr/bin/env bash
set -x
/opt/anaconda/bin/python3 create_input_files.py
/opt/anaconda/bin/python3 trtrain.py > log2.out
