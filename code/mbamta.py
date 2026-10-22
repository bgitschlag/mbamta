"""

Multi-taxon analysis of mutation-biased adaptation (mbamta)
using empirical and simulated data.

"""


# Import packages that will be needed for this script:
import os
import pandas as pd
import numpy as np
import mbamta_analyses as mba
import mbamta_figures as fig


# Open configuration file to execute input arguments:
args = {}
with open('./mbamta_config.txt') as f:
    exec(f.read(), {}, args)

# Number of datasets to simulate:
sim_n = int(args['simulations'])

# Substitutions attributed to non-adaptive mutations in robustness analyses:
contam = list(np.arange(0, float(args['contamination']), 0.1) + 0.1)
usedat = ['all_data'] + [
    f'{round(c * 100)}_percent_trimmed_data'
    for c in contam
]

# Correct for pseudoreplication due to shared
# selection conditions among species?
correct_for_pseudorep = args['correct_for_pseudorep']

# Define the measure of bias in mutation & adaptation data:
measures = ['rates', 'freqs']

# Overall data, adaptive data, & results directories:
ddir = './data'
adir = './data/adaptive_changes'
rslt = f'./results/results_with_stats_on_{sim_n}_sims'

# Species-specific mutation spectra
m_spec = pd.read_csv(f'./data/mutation_spectra.csv')

# Species names & mutation counts used to derive mutation spectra:
sp_mutn = pd.read_csv(
    './data/species_mutation_spectrum_counts.csv'
).sort_values(by='species', ignore_index=True)

# Taxonomic dependence (individualized & pairwise lists of species
# to omit in downstream analyses, to see if results are driven by a
# specific taxon or pseudoreplication via phylogenetic nonindependence):
taxon_ind = [[s] for s in list(sp_mutn['species'])]
phylo_ind = args['phylogenetic_independence']

# Types of mutation bias to search for in adaptation data:
btypes = [
    'transition',
    'base_mutability',
    'general',
    'alternate_base'
]

# Unique DNA bases & dictionary of base-pairing
bs = ['A', 'C', 'G', 'T']
bp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

# Column titles for mutation-vs-adaptation analysis output:
mvacol = [
    'species',
    'paths',
    'mu',
    'adaptive',
    'm_bias',
    'a_bias',
    'm_bcpath',
    'a_bcpath'
]

# The six unique single-nucleotide mutation classes
mutations = [f'{n}>{m}' for n in [bs[0], bs[2]]
             for m in bs if n != m]

# Color-coding on mutation classes, for figures
colors = {
    'A>C': 'xkcd:orange red',
    'A>G': 'xkcd:azure',
    'A>T': 'xkcd:pink purple',
    'G>A': 'xkcd:blue',
    'G>C': 'xkcd:orange',
    'G>T': 'xkcd:scarlet'
}

# Unicode symbols for graphing mutation classes per species
ucdsps = [
    u"\u25CF",
    u"\u25C6",
    u"\u25B2",
    u"\u25BC",
    u"\u25B6",
    u"\u25C0"
]

# Unicode symbols for figures (defaults to circles if n>20)
ucdsym = [
    u"\u25B7", u"\u25ED", u"\u25BC", u"\u25CF",
    u"\u25A1", u"\u25B3", u"\u25C7", u"\u25C1",
    u"\u25EF", u"\u25C6", u"\u25EE", u"\u25CA",
    u"\u25B2", u"\u25BD", u"\u25A0", u"\u25B6",
    u"\u25C0", u"\u25D0", u"\u25D1", u"\u25D2"
]

# Amino acid abbreviations
amino_acid = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D",
    "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
    "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
    "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"
}

# Dictionary specifying the standard genetic code
genetic_code = {
    'TTT': 'F', 'TCT': 'S', 'TAT': 'Y', 'TGT': 'C',
    'TTC': 'F', 'TCC': 'S', 'TAC': 'Y', 'TGC': 'C',
    'TTA': 'L', 'TCA': 'S', 'TAA': '*', 'TGA': '*',
    'TTG': 'L', 'TCG': 'S', 'TAG': '*', 'TGG': 'W',
    'CTT': 'L', 'CCT': 'P', 'CAT': 'H', 'CGT': 'R',
    'CTC': 'L', 'CCC': 'P', 'CAC': 'H', 'CGC': 'R',
    'CTA': 'L', 'CCA': 'P', 'CAA': 'Q', 'CGA': 'R',
    'CTG': 'L', 'CCG': 'P', 'CAG': 'Q', 'CGG': 'R',
    'ATT': 'I', 'ACT': 'T', 'AAT': 'N', 'AGT': 'S',
    'ATC': 'I', 'ACC': 'T', 'AAC': 'N', 'AGC': 'S',
    'ATA': 'I', 'ACA': 'T', 'AAA': 'K', 'AGA': 'R',
    'ATG': 'M', 'ACG': 'T', 'AAG': 'K', 'AGG': 'R',
    'GTT': 'V', 'GCT': 'A', 'GAT': 'D', 'GGT': 'G',
    'GTC': 'V', 'GCC': 'A', 'GAC': 'D', 'GGC': 'G',
    'GTA': 'V', 'GCA': 'A', 'GAA': 'E', 'GGA': 'G',
    'GTG': 'V', 'GCG': 'A', 'GAG': 'E', 'GGG': 'G'
}


# Function to run full set of analyses
# of mutation-biased adaptation for input parameters:
# measurement unit ("measure"; 'freqs' or 'rates'),
# data fraction ("data"; all or trimmed),
# modeled dynamics ("dyanmics"; 'mu' or 'mu_and_effects')
def mutation_biased_adaptation_analysis(
        rdir, measure, data, dynamics, u,
        pseudorep, correct_for_pseudorep
):

    if data != 'all_data':
        cntm = contam[usedat.index(data) - 1]
    else:
        cntm = 0

    if not os.path.exists(rdir):
        os.makedirs(rdir)

        mba.mva_compare(
            adir, rdir, measure, data, mutations, sp_mutn, sim_n,
            mvacol, m_spec, bp, bs, btypes, cntm, u, genetic_code,
            pd.DataFrame(mutations, columns=['mutation']),
            dynamics, pseudorep, correct_for_pseudorep
        )

    fig.summary_tables(f'{rdir}/analysis', measure, data, mutations, bp)

    fig.iter_sans_species(
        rdir, 'phylogenetic_nonindependence',
        phylo_ind, mutations
    )

    fig.iter_sans_species(
        rdir, 'sans_one_species',
        taxon_ind, mutations
    )

    fig.figures(
        rdir, measure, mutations, bp, sim_n, sp_mutn,
        btypes, data, ['adaptive', 'de_novo'], args['fd'],
        colors, ucdsps, ucdsym, taxon_ind, phylo_ind
    )


# Call the functions to analyze adaptive vs mutation
# data & graph the results:
for measure in measures:

    for data in usedat:

        mutation_biased_adaptation_analysis(
            f'{rslt}/bias_in_{measure}/{data}',
            measure, data, 'mu', 'N/A', 'N/A',
            correct_for_pseudorep=False
        )

        if measure == 'rates' and data == 'all_data':

            # Extract amino acid substitutions and U
            # (Universal amino acid exchangeability index)
            # from Table 2 of Tang et al. 2004:
            t_u = pd.read_csv(f"./data/{args['aa_effects']}")
            t_u = t_u.drop(t_u.index[-1])
            t_u['AA1'] = t_u['AA1'].replace(amino_acid)
            t_u['AA2'] = t_u['AA2'].replace(amino_acid)

            u = pd.DataFrame(
                zip(
                    [f"{t_u['AA1'][i]}>{t_u['AA2'][i]}>{t_u['AA1'][i]}"
                     for i in range(len(t_u))],
                    t_u['U (Universal EI)']
                ),
                columns=['aa', 'u']
            )

            # Run full analysis on 'all_data' while modeling
            # dynamics as a function of both mutation rates
            # and amino acid-level effects:
            mutation_biased_adaptation_analysis(
                f'{rslt}/bias_in_{measure}'
                '/dynamics_with_amino_acid_effects',
                measure, 'all_data', 'mu_and_effects',
                u, 'N/A', correct_for_pseudorep=False
            )

            if correct_for_pseudorep:
                # Analyze 'all_data' (no suspected contamination trimming),
                # edited to eliminate suspected pseudoreplicates due to
                # overlap of selection conditions across species:
                pseudorep = pd.DataFrame(
                    zip(
                        args['species'],
                        args['query_cat'],
                        args['criteria']
                    ), columns=['species', 'query_cat', 'criteria']
                )

                mutation_biased_adaptation_analysis(
                    f'{rslt}/bias_in_{measure}'
                    '/selection_pseudoreplicates_omitted',
                    measure, 'all_data',
                    'mu', 'N/A', pseudorep,
                    correct_for_pseudorep=True
                )

    # Plot effects of assuming varying levels of contamination
    # of the adaptive data by non-adaptive mutations:
    if len(usedat) > 1:
        fig.fig_contam_effect(measure, mutations, bp, sim_n, contam, colors)
