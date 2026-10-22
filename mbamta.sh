#!/usr/bin/env bash
set -euo pipefail

# Mutation-biased adaptation multi-taxon analysis (mbamta):
python code/mbamta.py

# Compare synonymous & adaptive biases with each other & with mutation:
python code/syn_adpt_mut_compare.py

# Compare mutation bias vs adaptive bias by drug or target:
python code/mtb_analysis_by_drug.py
