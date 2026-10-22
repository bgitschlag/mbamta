# mbamta
Mutation-biased adaptation, multi-taxon analysis

This document provides guidance on running mbamta, a computational pipeline implemented in Python for performing a Mutation-Biased Adaptation Multi-Taxon Analysis, using empirical data, with statistical analysis performed using a bootstrapping approach and the simulation of data under the null hypothesis in which mutational biases have no effect on the direction of adaptive evolution.

This software is equipped to be run from the command line:

```
bash mbamta.sh
```

The input arguments, e.g. number of simulations for bootstrapping and statistical significance and upper level of non-adaptive contamination to assess, are passed from the configuration file mbamta_config.txt to the Python files.

For a local computer, e.g. 4 to 8 CPU cores and 16GB RAM, allow approximately 8 to 12 minutes for script to run for n of 100; or 2 to 4 hours for n of 10k.

SOURCE DATA:

This program uses the following source data:


1. Raw adaptive amino acid substitutions from all included species, with nucleotide attributions:

```
data/adaptive_changes.csv
```

2. Original data files from the source literature on three species for which synonymous substitution rates are also used for analysis (E. coli, S. cerevisiae, and P. falciparum):

```
data/adaptive_changes/cowell_table_s4.csv
data/adaptive_changes/ottilie_data_s4.csv
data/adaptive_changes/tenaillon_table_s2.csv
```

3. Codon frequencies for all included species:

```
data/codon_use.csv
```

4. Original data files from source literature on adaptive substitutions in M. tuberculosis, together with a sequence file containing the nucleotide sequence for 16s rRNA gene rrs in M. tuberculosis:

```
data/m_tuberculosis/m_tuberculosis_rrs.txt
data/m_tuberculosis/manson-tableS5.csv
data/m_tuberculosis/pbio.3000265.s004.csv
```

5. Mutation rate spectra, derived from raw counts normalized to GC content (see Materials and methods), for all included species:

```
data/mutation_spectra.csv
```

6. Names and mutation counts (raw sample sizes underlying mutation rate spectra) for all included species:

```
data/species_mutation_spectrum_counts.csv
```

7. Amino acid exchangeability indices from source literature (Tang et al. MBE 2004) for the analysis that controls for selection on average amino acid-level effects:

```
data/tang_et_al_2004_table2.csv
```
