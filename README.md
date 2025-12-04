# mbamta
Mutation-biased adaptation, multi-taxon analysis

This document provides guidance on running mbamta.py, a Python script for performing a Mutation-Biased Adaptation Multi-Taxon Analysis, using empirical data, with statistical analysis performed using a bootstrapping approach and the simulation of data under the null hypothesis in which mutational biases have no effect on the direction of adaptive evolution.

This script is equipped to be run from the command line:

```
python3 /DIRECTORY/mbamta.py n
```

The argument n is an integer that specifies the number of simulated data sets to be generated and analyzed for the purpose of gathering statistics on the results. n=10 is recommended for testing purposes.

For a local computer, e.g. 4 to 8 CPU cores and 16GB RAM, allow approximately 8 to 12 minutes for script to run for n of 100; or 2 to 4 hours for n of 10k.


SOURCE DATA:

This program uses four types of source data, each of which is species-specific: the spectra of new mutations and the sample sizes for the mutation-rate measurements used to generate these mutation spectra, the codon-usage frequencies that will serve as weights on the mutation-rate measurements to derive expectations for protein-coding changes, and spectra of raw adaptive substitutions. (Example input data can be downloaded from https://github.com/bgitschlag/mbamta/blob/main/Gitschlag_et_al_2025_SOURCE_DATA.zip):

Taxa included in analysis (one column of the following CSV file):
```
./SOURCE_DATA/species_list_and_mutation_counts.csv
```

Sample sizes for mutation-rate measurements (one column of the following CSV file):
```
./SOURCE_DATA/species_list_and_mutation_counts.csv
```

Mutation spectra (GC-weighted and normalized to sum to 1):
```
./SOURCE_DATA/mutation_spectra.csv
```

Codon-usage frequency tables, species-specific:
```
./SOURCE_DATA/codon_use/{species}.csv
```

Adaptive observations (includes a column of raw counts):
```
./SOURCE_DATA/adaptive_changes/adaptive_csv/{species}_adaptive_changes.csv
```
