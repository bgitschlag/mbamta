# Mutation-Biased Adaptation Multi-Taxon Analysis. This script will evaluate
# the putative effect of mutational biases on adaptive outcome, using
# empirical data, with statistical analysis performed using a bootstrapping
# approach and simulated data under the null hypothesis in which mutation bias
# has no effect. See READ ME file for more information and guidance.

# Import packages that will be needed for this script:
import sys
import os
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.lines as lines


# Specify number of data sets to be simulated for statistical confidence
# and significance; specify amounts of the adaptive substitutions data that
# will be attributed to non-adaptive mutations, for robustness analysis:
sim_n, assumed_contam = int(sys.argv[1]), [0.1, 0.2, 0.3, 0.4]

# Specify amount of adaptive substitutions data to be used for analyses:
usedata = ['all_data'] + [f'{round(c * 100)}_percent_trimmed_data'
                          for c in assumed_contam]

# Specify directory containing the adaptive substitutions data:
adir = './published_data/adaptive_changes/adaptive_csv'

# Create DataFrame of species-specific mutation spectra by reading CSV file:
m_spec = pd.read_csv('./published_data/mutation_spectra.csv')

# Create DataFrame of species names and mutation counts (the number
# of empirical observations that inform the mutation-rate spectrum),
# by reading CSV file; order species alphabetically:
sp_mutn = pd.read_csv('./published_data/species_list_and_mutation_counts'
                      '.csv').sort_values(by='species', ignore_index=True)

# Specify types of mutation bias that will be used in downstream analyses:
btypes = ['transition', 'base_mutability', 'general', 'alternate_base']

# Define unique DNA bases; create dictionary that specifies the base-pairing:
bs, bp = ['A', 'C', 'G', 'T'], {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

# Define output table column titles for mutation-vs-adaptation analysis:
mvacol = ['species', 'paths', 'genome_mu', 'mu', 'adaptive',
          'mu_bias', 'a_bias', 'mu_bcpath', 'a_bcpath']

# Define the six unique single-nucleotide mutation classes:
mutations = [f'{n}>{m}' for n in [bs[0], bs[2]] for m in bs if n != m]

# Color-coding on single-nucleotide mutation classes, for graphing purposes:
colors = {'A>C': 'xkcd:orange red', 'A>G': 'xkcd:blue',
          'A>T': 'xkcd:pink purple', 'G>A': 'xkcd:azure',
          'G>C': 'xkcd:orange', 'G>T': 'xkcd:scarlet'}

# Unicode symbols for graphing single-nucleotide mutation classes per species:
ucdsps = [u"\u25CF", u"\u25C6", u"\u25B2", u"\u25BC", u"\u25B6", u"\u25C0"]

# Unicode symbols for graphing data (code defaults to circles if n>20):
ucdsym = [u"\u25B7", u"\u25ED", u"\u25BC", u"\u25CF", u"\u25A1",
          u"\u25B3", u"\u25C7", u"\u25C1", u"\u25EF", u"\u25C6",
          u"\u25EE", u"\u25CA", u"\u25B2", u"\u25BD", u"\u25A0",
          u"\u25B6", u"\u25C0", u"\u25D0", u"\u25D1", u"\u25D2"]

# Create dictionary specifying the standard genetic code:
genetic_code = {'TTT': 'F', 'TCT': 'S', 'TAT': 'Y', 'TGT': 'C',
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
                'GTG': 'V', 'GCG': 'A', 'GAG': 'E', 'GGG': 'G'}


# Create folder with specified subfolders, for saving output files:
def folder(folders, subfolders):
    for f in folders:
        [os.makedirs(f + s) for s in subfolders if not os.path.exists(f + s)]
    return folders


# Append a row to a DataFrame:
def merge(append_to, appendix):
    if append_to.empty:
        append_to = appendix
    else:
        append_to = pd.concat([append_to, appendix])
    return append_to


# Format numbers for visual presentation on graphs:
def num_form(nums):
    f = []
    for n in nums:
        if abs(n) < 0.01:
            f.append(np.format_float_scientific(n, precision=1, min_digits=1))
        else:
            f.append(round(n, 2))
    return f


# Reformat species names for graphing purposes:
def species_name(species):
    return species[0].upper() + f". {species[2:]}"


# Create species-specific dictionary of codon usage frequencies:
def codon_use(species):
    ctbl, cfrq = pd.read_csv(f'./published_data/codon_use/{species}.csv'), {}
    for codon in range(len(ctbl)):
        cfrq[ctbl.iloc[codon]['codon']] = (ctbl.iloc[codon]['fraction'])
    return cfrq


# Trim an adaptive substitutions spectrum (emp) by removing
# a fraction with a neutral mutation spectrum (mspec)
def trim(mspec, emp):

    def trimmed_error(x):
        est = (mspec * cntm) + (np.array([x[0], x[1], x[2], x[3],
                                          x[4], x[5]]) * (1 - cntm))
        ssq = sum((emp - est) ** 2)
        return ssq

    m = minimize(trimmed_error, np.ones(6) / 6, method="Nelder-Mead",
                 bounds=np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                                  [0, 1]]), options={"adaptive": True}
                 )
    aspec = np.array([m['x'][i] for i in range(6)])

    # Return normalized adaptive spectrum:
    return aspec / sum(aspec)


# Generate spectrum of available missense mutation paths, to be used as the
# null model for the adaptive substitutions spectrum and as the weights
# on the mutation spectrum to obtain the missense-specific mutation spectrum:
def path_spectrum(code, bp, bs, codon_use, counts, mut_quant, spectrum):
    tot_paths = 0
    # Iterate across all codons:
    for codon in code:
        # Only use amino acid codons (exclude stop codons):
        if code[codon] != '*':
            paths_from_codon = np.zeros(6)
            # Evaluate mutations for each codon position:
            for position in range(3):
                alt_bs = [n for n in bs if n != codon[position]]
                # Evaluate all alternative alleles for each codon position:
                for alt_base in alt_bs:
                    mutation, cdn = '', list(codon)
                    cdn[position] = alt_base
                    mutant = mutation.join(cdn)
                    # Make sure the mutation changes an amino acid
                    # but does not make a stop codon:
                    if code[mutant] != '*' and code[codon] != code[mutant]:
                        tot_paths += 1
                        ref, alt = codon[position], alt_base
                        if codon[position] in [bs[1], bs[3]]:
                            ref, alt = bp[ref], bp[alt]
                        paths_from_codon[mutations.index(f'{ref}>{alt}')] += 1
                        counts[mutations.index(f'{ref}>{alt}')] += 1
            # Take product of mutation paths from codon & codon-use fraction
            # & add to an array:
            mut_quant += codon_use[codon] * paths_from_codon
    # Normalize:
    mut_quant = mut_quant / sum(mut_quant)
    spectrum['code_freq'] = list(counts / sum(counts))
    spectrum['paths'] = mut_quant
    return spectrum


# Specify the contribution, by species and mutation class,
# toward available mutational paths, de novo mutations,
# adaptive substitutions, and associated biases:
def m_id_cont(mutation_in, mutation_out):
    mutation_out[0] += mutation_in['paths']
    mutation_out[1] += mutation_in['genome_mu']
    mutation_out[2] += mutation_in['mu']
    mutation_out[3] += mutation_in['adaptive']
    if len(mutation_out) == 8:
        mutation_out[4] += mutation_in['mu_bias']
        mutation_out[5] += mutation_in['a_bias']
        mutation_out[6] += mutation_in['mu_bcpath']
        mutation_out[7] += mutation_in['a_bcpath']
    return mutation_out


# Create list of numbers of adaptive substitution events
# per mutation class in a given species:
def adaptive_counter(species, counts):

    data = pd.read_csv(adir + f'/{species}_adaptive_changes.csv')
    for m in range(len(data)):
        if data.iloc[m]['ref_base'] in [bs[0], bs[2]]:
            r_n = data.iloc[m]['ref_base']
            a_n = data.iloc[m]['alt_base']
        else:
            r_n = bp[data.iloc[m]['ref_base']]
            a_n = bp[data.iloc[m]['alt_base']]
        counts[mutations.index(f"{r_n}>{a_n}")] += int(data.iloc[m]['events'])

    # Return list of counts of adaptive events by mutation class:
    return counts


# Calculate & save missense-specific mutation spectrum, adaptive substitutions
# spectrum, available mutational path spectrum, and associated biases:
def adaptations_per_species(sdir, s, d, genome_mu, adata, c):
    spm = pd.DataFrame(zip(mutations), columns=['mutation'])
    ps = path_spectrum(c, bp, bs, codon_use(s), np.zeros(6), np.zeros(6), spm)
    mut_n = int(sp_mutn.iloc[[list(sp_mutn['species']).index(s)]]
                ['mu_counts'].iloc[0])
    u = ((np.array(genome_mu[s]) * mut_n) + 1) * np.array(ps['paths'])
    u = u / sum(u)
    spm['genome_mu'], spm['paths'], spm['mu'] = genome_mu[s], ps['paths'], u
    a_n = np.array(adata)
    if d != 'all_data':
        a_n = trim(u, a_n / sum(a_n)) * (sum(np.array(adata)) * (1 - cntm))
    spm['adaptive_n'] = a_n
    a_n += 1
    a = a_n / sum(a_n)
    spm['adaptive'] = a
    spm['mu_bias'], spm['a_bias'] = u / (1 - u), a / (1 - a)
    bcps = [np.concatenate((x[:3] / (sum(x[:3]) - x[:3]),
                            x[3:] / (sum(x[3:]) - x[3:]))) for x in [u, a]]
    spm['mu_bcpath'], spm['a_bcpath'] = np.array(bcps[0]), np.array(bcps[1])
    spm.to_csv(sdir + f'/{s}_outcomes_model_vs_empirical_{d}.csv', index=False)
    print(f"spectra of mutations and adaptive changes for {s} saved to folder")


# Save output files of spectra and bias values by mutation type:
def adp_per_mut_type(sdir, mdir, data, template, col):
    for m in range(len(template)):
        m_v_a = pd.DataFrame(columns=col)
        ref = template.iloc[m]['mutation'][0]
        alt = template.iloc[m]['mutation'][-1]
        for s in sp_mutn['species']:
            mvad = pd.read_csv(sdir +
                               f'/{s}_outcomes_model_vs_empirical_{data}.csv')
            for n in range(len(mvad)):
                mi = mvad.iloc[n]
                if mi['mutation'][0] == ref and mi['mutation'][-1] == alt:
                    mva = pd.DataFrame([[s, float(mi['paths']),
                                         float(mi['genome_mu']),
                                         float(mi['mu']),
                                         float(mi['adaptive']),
                                         float(mi['mu_bias']),
                                         float(mi['a_bias']),
                                         float(mi['mu_bcpath']),
                                         float(mi['a_bcpath'])]],
                                       columns=col
                                       )
                    m_v_a = merge(m_v_a, mva)
        m_v_a.to_csv(mdir + f'/mutation_vs_outcomes_for_{ref}>{alt}.csv',
                     index=False)
    return print(f"frequencies of de novo & adaptive mutations "
                 f"of each type saved to folder")


# Overall contribution of all available data toward rates, frequencies, and
# biases, for transition-transversion ratio and by individual mutation class:
def mcat_cont(species, data, all_mutations, tb, sdir):

    for s in species:
        ma = pd.read_csv(sdir + f'/{s}_outcomes_model_vs_empirical_{data}.csv')
        ti, tv = [0, 0, 0, 0], [0, 0, 0, 0]
        for n in range(len(ma)):
            mi = ma.iloc[n]
            ref, alt = mi['mutation'][0], mi['mutation'][-1]
            mdata = pd.DataFrame([[s, f'{ref}>{alt}'] +
                                  m_id_cont(mi, [0, 0, 0, 0, 0, 0, 0, 0])],
                                 columns=[mvacol[0], 'mutation'] + mvacol[1:]
                                 )
            all_mutations = merge(all_mutations, mdata)
            if alt in [bs[0], bs[2]]:
                ti = m_id_cont(mi, ti)
            else:
                tv = m_id_cont(mi, tv)
        tb = merge(tb, pd.DataFrame([[s] + list(np.array(ti) / np.array(tv))],
                                    columns=mvacol[:3] + mvacol[5:7]))

    # Return DataFrames of transition bias and rates, frequencies & biases
    # for all six single-nucleotide mutation classes:
    return tb, all_mutations


# Calculate the log likelihood for a pairwise set of actual & expected data:
def log_likelihood(expected, actual, size):
    ssq = sum((expected - actual) ** 2)
    var = ssq / size
    return (size / 2) * (-np.log(2 * np.pi) - np.log(var) - 1)


# Calculate log likelihood by using a regression model to generate expected
# adaptive substitution biases and compare them against the actual
# (empirical or simulated) adaptive substitution biases:
def model_likelihood(mutations, adaptations, slope, intercepts, log_l):
    for m in range(len(mutations)):
        reg = slope * np.array(mutations[m]) + np.array(intercepts[m])
        log_l += log_likelihood(np.array(reg), np.array(adaptations[m]),
                                len(adaptations[m]))
    return log_l


# Calculate regression details (slope, intercept, Pearson's r)
# for the (x,y) coordinates of a data set:
def least_sq(x, y):
    ave_x, ave_y = sum(x) / len(x), sum(y) / len(y)
    cov_xy = (sum(x * y) / len(x)) - (ave_x * ave_y)
    var_x = (sum(x ** 2) / len(x)) - (ave_x ** 2)
    var_y = (sum(y ** 2) / len(y)) - (ave_y ** 2)
    if var_y > 0:
        r = cov_xy / np.sqrt(var_x * var_y)
    else:
        r = "N/A"
    slope = cov_xy / var_x
    return [slope, ave_y - (slope * ave_x), r]


# Use a maximum-likelihood procedure to estimate the shared slope (beta) and
# intercepts (beta_0) for the six single-nucleotide mutation classes:
def max_likelihood(expected, observed, initial_intercepts):
    itl = np.concatenate((np.array([0]), initial_intercepts))

    def neg_l(x):
        return -model_likelihood(expected, observed, x[0],
                                 [x[1], x[2], x[3], x[4], x[5], x[6]], 0)

    m = minimize(neg_l, itl, method="Nelder-Mead", options={"adaptive": True})
    return [m['x'][i] for i in range(7)]


# Calculate & save details of confidence estimates (top, bottom,
# and midpoint of 95% bootstrap confidence interval):
def confidence(rsdir, stdir, filep, stcol):
    if not os.path.exists(stdir):
        os.makedirs(stdir)
    var, val, mdp, err, top, btm = [], [], [], [], [], []
    for (columnName, columnData) in pd.read_csv(rsdir + f'/{filep}',
                                                encoding='utf-8').items():
        if columnName != 'data':
            var.append(columnName)
            val.append(columnData.values[0])
            bs = np.sort(np.array(columnData[1:]))
            lo = bs[round(0.025 * len(bs))]
            hi = bs[round(0.975 * len(bs)) - 1]
            error = (hi - lo) / 2
            mdp.append(lo + error)
            err.append(error)
            top.append(hi)
            btm.append(lo)
    pd.DataFrame(np.array([var, val, mdp, err, top, btm]).T,
                 columns=stcol).to_csv(stdir + f'/{filep}')


# Perform and save statistics results across the simulated data sets:
def bootstats(rsdir, mdl, stcol, spec):
    rd, std = 'bootstrap_results', 'statistics'
    for root, dirs, files in os.walk(rsdir, topdown=True):
        stdir = root[:root.index(rd)] + std + root[root.index(rd) + len(rd):]
        for file in files:
            if file != '.DS_Store':
                confidence(root, stdir, file, stcol)
    print(f"Bootstrap statistics saved for {spec} {mdl} results")


# Generate a simulated data set by resampling from a specified input data set:
def bootstrap(data, dsiz):
    boot = np.array([np.array([np.ones(6) for s in range(len(sp_mutn))])
                     for r in range(sim_n)])
    for r in range(sim_n):
        for s in range(len(sp_mutn)):
            u, c = np.unique(np.random.choice(6, size=dsiz[s], replace=True,
                                              p=np.array(list(data[s] /
                                                              sum(data[s])))),
                             return_counts=True)
            np.put(boot[r][s], u, c + 1)
            boot[r][s] /= sum(boot[r][s])
    return boot


# Calculate various measures of bias from spectrum data:
def biasmeasures(data):
    g = sum(data[3:])
    return (data[1] + data[3]) / (data[0] + data[2] +
                                  data[4] + data[5]), data / (1 - data), \
           g / (1 - g), np.concatenate((data[:3] / (1 - g - data[:3]),
                                        data[3:] / (g - data[3:])))


# Save regression results for simulated and empirical data sets:
def bootresults(bdir, filename, empirical, simdata, col):
    tbl = pd.DataFrame(empirical + simdata, columns=col)
    tbl.insert(0, 'data', pd.Series(['empirical'] + [f'bootstrap_{r+1}'
                                                     for r in range(sim_n)]))
    tbl.to_csv(bdir + filename, encoding='utf-8', index=False)
    del tbl


# Perform regressions on a per-mutation-class basis for the six
# single-nucleotide classes, empirical and simulated data:
def pmbias(bdir, bias, loga, logm, boot, lboot, dsrc, btyp, bi):
    e = [[list(bias[bi][i]) + least_sq(logm[i], loga[i])] for i in range(6)]
    if dsrc == "adaptive":
        n = [[list(boot[bi].T[r].T[i]) + least_sq(logm[i], lboot[i].T[r])
              for r in range(sim_n)] for i in range(6)]
    else:
        n = [[list(boot[bi].T[r].T[i]) + least_sq(lboot[i].T[r], loga[i])
              for r in range(sim_n)] for i in range(6)]
    for i in range(6):
        bootresults(bdir, f'/{mutations[i]}_{btyp}_bias_{dsrc}.csv',
                    e[i], n[i], list(sp_mutn['species']) +
                    ['slope', 'intercept', 'r']
                    )
    del e, n


# Perform regressions on consolidated data formats (transition bias and
# base-mutability bias), empirical and simulated data:
def cbias(bdir, bias, loga, logm, boot, lboot, dsrc, btyp, bi):
    e = [list(bias[bi]) + least_sq(logm, loga)]
    if dsrc == "adaptive":
        n = [list(boot[bi].T[r]) + least_sq(logm, lboot[r])
             for r in range(sim_n)]
    else:
        n = [list(boot[bi].T[r]) + least_sq(lboot[r], loga)
             for r in range(sim_n)]
    bootresults(bdir, f'/{btyp}_bias_{dsrc}.csv', e, n,
                list(sp_mutn['species']) + ['slope', 'intercept', 'r'])
    del e, n


# Perform single regression aggregating across intra- and interspecies
# variation in bias, empirical and simulated data:
def ixibias(bdir, loga, logm, lboot, dsrc):
    e = [least_sq(logm.flatten(), loga.flatten())]
    if dsrc == "adaptive":
        n = [least_sq(logm.flatten(), lboot.T[r].T.flatten())
             for r in range(sim_n)]
    else:
        n = [least_sq(lboot.T[r].T.flatten(), loga.flatten())
             for r in range(sim_n)]
    bootresults(bdir, f'/inter_x_intra_species_bias_{dsrc}.csv', e, n,
                ['slope', 'intercept', 'r'])
    del e, n


# Perform regressions across mutation types,
# i.e. intra-species adaptive vs mutational bias:
def intraspbias(bdir, mub, adb, logm, loga, boot, lboot, dsrc):
    for s in range(len(sp_mutn)):
        sp = list(sp_mutn['species'])[s]
        if dsrc == "adaptive":
            e = [list(adb[s]) + least_sq(logm[s], loga[s])]
            n = [list(boot[r][s]) + least_sq(logm[s], lboot[r][s])
                 for r in range(sim_n)]
        else:
            e = [list(mub[s]) + least_sq(logm[s], loga[s])]
            n = [list(boot[r][s]) + least_sq(lboot[r][s], loga[s])
                 for r in range(sim_n)]
        bootresults(bdir, f'/{sp}_intra_sp_bias_{dsrc}.csv', e, n,
                    mutations + ['slope', 'intercept', 'r'])
        del e, n


# Perform regression using maximum-likelihood methodology to isolate
# interspecies variation in bias, with per-mutation proportionality
# constants, empirical and simulated data:
def ipmbias(bdir, bias, loga, logm, boot, lboot, dsrc, btyp, bi, altpath):
    itl_icp = np.log10(np.ones(6) / altpath)
    ml, n = max_likelihood(logm, loga, itl_icp), [[], [], [], [], [], []]
    e = [[list(bias[bi][i - 1]) + [ml[0], ml[i]]] for i in range(1, 7)]
    if dsrc == "adaptive":
        for r in range(sim_n):
            ml_r = max_likelihood(logm, lboot[r].T, itl_icp)
            [n[i-1].append(list(boot[bi].T[r].T[i-1]) + [ml_r[0], ml_r[i]])
             for i in range(1, 7)]
    else:
        for r in range(sim_n):
            ml_r = max_likelihood(lboot[r].T, loga, itl_icp)
            [n[i-1].append(list(boot[bi].T[r].T[i-1]) + [ml_r[0], ml_r[i]])
             for i in range(1, 7)]
    for i in range(6):
        bootresults(bdir,
                    f'/{mutations[i]}_intersp_panmut_{btyp}_bias_{dsrc}.csv',
                    e[i], n[i], list(sp_mutn['species']) +
                    ['slope', 'intercept']
                    )
    del e, n


# Analyze mutation vs adaptation data on bias, using empirical data
# and by simulating data:
def mva_analysis(rdir, model, data):

    ndir = rdir + '/analysis'
    scol = ['var', 'val', 'mid', 'err', '+95%', '-95%']
    adcol = {'model': 'adaptive', 'null': 'paths'}
    mdat, adat = np.empty((0, 6), int), np.empty((0, 6), int)
    rawdbsrc = np.empty((0, 6), int)
    ad_n = np.empty(0, int)
    for sp in list(sp_mutn['species']):
        spdat = pd.read_csv(rdir + f'/by_species'
                                   f'/{sp}_outcomes_model_vs_empirical_{data}'
                                   f'.csv'
                            )
        rawad = np.array(list(spdat['adaptive_n']))
        ad_n = np.append(ad_n, np.array([round(sum(rawad))]))
        mdat = np.append(mdat, np.array([list(spdat['mu'])]), axis=0)
        adat = np.append(adat, np.array([list(spdat[adcol[model]])]), axis=0)

        # Specify data source for generating simulated data
        # (empirical adaptive spectra or available mutational
        # paths for hypothesized model or null model, respectively):
        if model == 'model':
            rawdbsrc = np.append(rawdbsrc,
                                 np.array([rawad / sum(rawad)]), axis=0)
        else:
            nulldat = np.array(list(spdat['paths']))
            rawdbsrc = np.append(rawdbsrc,
                                 np.array([nulldat / sum(nulldat)]), axis=0)

    # Generate output folders and log-transformed data on mutation
    # and adaptive substitutions biases:
    for bt in btypes:
        folder([ndir + f'/{model}/bootstrap_results'],
               [f'/adaptive/{bt}_bias', f'/de_novo/{bt}_bias',
                '/adaptive/within_species', '/de_novo/within_species']
               )
    bdir = ndir + f'/{model}/bootstrap_results'
    mdb, adb = biasmeasures(mdat.T), biasmeasures(adat.T)
    lmti, lmgb = np.log10(mdb[0]), np.log10(mdb[1])
    lmbm, lmbc = np.log10(mdb[2]), np.log10(mdb[3])
    lati, lagb = np.log10(adb[0]), np.log10(adb[1])
    labm, labc = np.log10(adb[2]), np.log10(adb[3])

    # Simulate data sets and perform regression analyses:
    sim_source_data = [["adaptive", rawdbsrc, ad_n, adb],
                       ["de_novo", mdat, list(sp_mutn['mu_counts']), mdb]]
    for src in sim_source_data:
        bt = biasmeasures(bootstrap(src[1], src[2]).T)
        btti, btgb = np.log10(bt[0].T), np.log10(bt[1])
        btbm, btbc = np.log10(bt[2].T), np.log10(bt[3])
        intraspbias(bdir + f'/{src[0]}/within_species', mdb[1].T, adb[1].T,
                    lmgb.T, lagb.T, bt[1].T, btgb.T, src[0])
        ixibias(bdir + f'/{src[0]}/general_bias', lagb, lmgb, btgb, src[0])
        cbias(bdir + f'/{src[0]}/transition_bias', src[3], lati, lmti,
              bt, btti, src[0], "transition", 0)
        cbias(bdir + f'/{src[0]}/base_mutability_bias', src[3], labm, lmbm,
              bt, btbm, src[0], "base_mutability", 2)
        pmbias(bdir + f'/{src[0]}/general_bias', src[3], lagb, lmgb, bt,
               btgb, src[0], "general", 1)
        ipmbias(bdir + f'/{src[0]}/general_bias', src[3], lagb, lmgb, bt,
                btgb.T, src[0], "general", 1, 5)
        pmbias(bdir + f'/{src[0]}/alternate_base_bias', src[3], labc,
               lmbc, bt, btbc, src[0], "alternate_base", 3)
        ipmbias(bdir + f'/{src[0]}/alternate_base_bias', src[3], labc,
                lmbc, bt, btbc.T, src[0], "alternate_base", 3, 2)
        del bt, btti, btgb, btbm, btbc

    # Generate and save statistical summary of regression results:
    bootstats(bdir + '/adaptive', model, scol, 'adaptive')
    bootstats(bdir + '/de_novo', model, scol, 'de_novo')
    print(f"Results from analysis of mutations vs adaptive "
          f"outcomes saved to folder")


# Generate and save output tables containing original on frequencies & bias
# measurements for mutation and adaptive substitution data, plus detailed
# regression analyses for hypothesized vs null model:
def mva_compare(adir, rdir, data, code, adata):
    rdir = folder([rdir], ['/by_mutation', '/by_species', '/analysis'])
    mdir, sdir = rdir[0] + '/by_mutation', rdir[0] + '/by_species'
    for s in sp_mutn['species']:
        adata[s] = adaptive_counter(s, [0, 0, 0, 0, 0, 0])
        adaptations_per_species(sdir, s, data, m_spec, adata[s], code)
    adata.to_csv(adir + f'/adaptive_counts_{data}.csv', index=False)
    adp_per_mut_type(sdir, mdir, data,
                     pd.DataFrame(mutations, columns=['mutation']), mvacol)
    c = mcat_cont(sp_mutn['species'], data,
                  pd.DataFrame(columns=[mvacol[0], 'mutation'] + mvacol[1:]),
                  pd.DataFrame(columns=mvacol), sdir
                  )
    ti_bias, all_mutations = c[0], c[1]
    ti_bias.to_csv(mdir + '/ti_bias_mutation_vs_outcomes.csv', index=False)
    all_mutations.to_csv(mdir + '/all_mutation_classes_vs_outcomes.csv',
                         index=False)
    for mdl in ['model', 'null']:
        mva_analysis(rdir[0], mdl, data)
    return print(f"RESULTS NOW COMPILED FOR {data}")


# Calculate regression significance (p-value) by comparing the empirical point
# estimate slope (beta) to the results for simulated data under the null model:
def rsig_simdat(empirical, simulated):
    overlap = 0
    for s in simulated:
        if s >= empirical:
            overlap += 1
    if overlap == 0:
        return f'<{1 / sim_n}'
    else:
        return f'{overlap / sim_n}'


# Generate and save table of summary statistics on regression analyses:
def summary_table(ardir, data):
    sdir = ardir + '/model/statistics/adaptive'
    nsdr = ardir + '/null/statistics/adaptive'
    biases = ['transition',
              'inter_x_intra_species'] + [f'{m}_general' for m in mutations]
    subd = ['transition'] + ['general' for i in range(7)]
    lbls = ['Transition',
            'Intra- & inter-species'] + [f'{m[0]}:{bp[m[0]]}>{m[2]}:{bp[m[2]]}'
                                         for m in mutations]
    islp, nslp, iicp, ir, isig = [], [], [], [], []
    pslp, npsp = ['N/A', 'N/A'], ['N/A', 'N/A']
    picp, psig = ['N/A', 'N/A'], ['N/A', 'N/A']
    for bias in biases:
        isomfl = f'/{subd[biases.index(bias)]}_bias/{bias}_bias_adaptive.csv'
        if biases.index(bias) >= 2:
            pan = pd.read_csv(sdir +
                              f'/general_bias/{bias[0:3]}'
                              f'_intersp_panmut_general_bias_adaptive.csv'
                              )
            pnn = pd.read_csv(nsdr +
                              f'/general_bias/{bias[0:3]}'
                              f'_intersp_panmut_general_bias_adaptive.csv'
                              )
            pnl = np.array(list(pd.read_csv(f'./results'
                                            f'/results_with_stats_on_{sim_n}'
                                            f'_sims/{data}/analysis/null'
                                            f'/bootstrap_results/adaptive'
                                            f'/general_bias'
                                            f'/{bias[0:3]}_intersp_panmut_'
                                            f'general_bias_adaptive.csv'
                                            )['slope']
                                )
                           )[1:]
            nsi = [i for i in range(len(pnn))
                   if 'slope' in list(pnn['var'])[i]][0]
            psi = [i for i in range(len(pan))
                   if 'slope' in list(pan['var'])[i]][0]
            pii = [i for i in range(len(pan))
                   if 'intercept' in list(pan['var'])[i]][0]
            pst = num_form([list(pan['val'])[psi], list(pan['val'])[pii]] +
                           [list(pan['-95%'])[psi], list(pan['+95%'])[psi]] +
                           [list(pan['-95%'])[pii], list(pan['+95%'])[pii]]
                           )
            nps = num_form([list(pnn['val'])[nsi], list(pnn['-95%'])[nsi],
                            list(pnn['+95%'])[nsi]])
            npsp.append(f'{nps[0]} ({nps[1]} \u2014 {nps[2]})')
            pslp.append(f'{pst[0]} ({pst[2]} \u2014 {pst[3]})')
            picp.append(f'{pst[1]} ({pst[4]} \u2014 {pst[5]})')
            psig.append(rsig_simdat(list(pan['val'])[psi], pnl))
        isn = pd.read_csv(nsdr + isomfl)
        iso = pd.read_csv(sdir + isomfl)
        nul = np.array(list(pd.read_csv(f'./results/'
                                        f'results_with_stats_on_{sim_n}_sims'
                                        f'/{data}/analysis/null'
                                        f'/bootstrap_results/adaptive'
                                        + isomfl)['slope']
                            )
                       )[1:]
        ni = list(iso['var']).index('slope')
        si = list(iso['var']).index('slope')
        ii = list(iso['var']).index('intercept')
        ri = list(iso['var']).index('r')
        ist = num_form([list(iso['val'])[si], list(iso['val'])[ii],
                        list(iso['val'])[ri]] +
                       [list(iso['-95%'])[si], list(iso['+95%'])[si]] +
                       [list(iso['-95%'])[ii], list(iso['+95%'])[ii]] +
                       [list(iso['-95%'])[ri], list(iso['+95%'])[ri]]
                       )
        nst = num_form([list(isn['val'])[ni], list(isn['-95%'])[ni],
                        list(isn['+95%'])[ni]])
        nslp.append(f'{nst[0]} ({nst[1]} \u2014 {nst[2]})')
        islp.append(f'{ist[0]} ({ist[3]} \u2014 {ist[4]})')
        iicp.append(f'{ist[1]} ({ist[5]} \u2014 {ist[6]})')
        ir.append(f'{ist[2]} ({ist[7]} \u2014 {ist[8]})')
        isig.append(rsig_simdat(list(iso['val'])[si], nul))
    pd.DataFrame(zip(lbls, islp, nslp, iicp, ir, isig, pslp, npsp, picp, psig),
                 columns=['Type of bias', 'beta', 'null beta', 'beta_0', 'r',
                          'significance', 'panmut_beta', 'panmut_null_beta',
                          'panmut_beta_0', 'panmut_significance'
                          ]
                 ).to_csv(sdir + '/regression_models_summary_stats.csv')


# Generate table of summary statistics for assessing effects of
# neutral contamination in adaptive substitutions data:
def sumstats(iso, nul, effl, tbl, cols):
    si = list(iso['var']).index(effl)
    pfrm, ptxt = 0, rsig_simdat(list(iso['val'])[si], nul)
    if '<' in ptxt:
        pfrm += 1
    tbl = merge(tbl, pd.DataFrame(zip([list(iso['val'])[si]],
                                      [list(iso['mid'])[si]],
                                      [list(iso['err'])[si]],
                                      [float(ptxt[pfrm:])]
                                      ),
                                  columns=cols)
                )
    return tbl


# Specify customized labels and location of axis tic-marks
# depending on the distribution of data values:
def axis_vals(minobs, maxobs, expand):

    logmin, logmax = np.floor(np.log10(minobs)), np.ceil(np.log10(maxobs))
    rotate, xmk = 0, 10 ** np.arange(logmin, logmax + 1)
    bl = [float(m) for m in xmk if m < 1]
    ab = [int(round(m)) for m in xmk if m >= 1]
    mlb = bl + ab
    if expand == True:
        if minobs >= (xmk[0] * 3):
            xmk[0], mlb = xmk[0] * 3, mlb[1:]
            if maxobs > (xmk[-1] * 0.3):
                xmk[-1] = xmk[-1] * 1.3
            else:
                xmk[-1], mlb = xmk[-1] * 0.3, mlb[:-1]
    if xmk[0] <= 0.0001:
        rotate += 45

    # Return axis marker labels, axis tic-mark values, and
    # angular rotation (0 or 45 degrees) on axis tic labels:
    return mlb, xmk, rotate


# Calculate minimum, maximum, and range of axis values,
# for purposes of positioning labels on graph:
def axmeasures(axv):
    axmin, axmax = np.log10(axv[1])[0], np.log10(axv[1])[-1]
    return axmin, axmax, axmax - axmin


# Iterate through the species list, leaving one species out at a time
# while performing regression analyses on the remaining data:
def iter_sans_species(rdir):
    sspdir, mdir = rdir + f'/sans_one_species', rdir + '/by_mutation'
    if not os.path.exists(sspdir):
        os.makedirs(sspdir)
        files = ['all_mutation_classes_vs_outcomes.csv',
                 'ti_bias_mutation_vs_outcomes.csv'
                 ] + [f'mutation_vs_outcomes_for_{m}.csv' for m in mutations]
        snfl = ['all_snm_bias_adaptive.csv',
                 'transition_bias_adaptive.csv'] + [f'{m}_bias_adaptive.csv'
                                                    for m in mutations]
        col = ['dataset', 'slope', 'intercept', 'r']
        species = list(sp_mutn['species'])
        sspres = [pd.DataFrame(columns=col) for i in range(8)]
        ipmbias = [pd.DataFrame(columns=col[:-1]) for i in range(6)]
        for sans in species:
            mspr, aspr = [], []
            for file in files:
                srcd = pd.read_csv(rdir + f'/by_mutation/{file}')
                udat = pd.DataFrame()
                for i in srcd.index:
                    if srcd.loc[i]['species'] != sans:
                        udat = merge(udat, srcd.loc[i].to_frame().T)
                mb = np.log10(np.array(list(udat['mu_bias'])))
                ab = np.log10(np.array(list(udat['a_bias'])))
                if files.index(file) >= 2:
                    mspr.append(mb)
                    aspr.append(ab)
                ls = least_sq(mb, ab)
                lsrow = pd.DataFrame(zip([f'sans_{sans}'], [ls[0]],
                                         [ls[1]], [ls[2]]), columns=col)
                sspres[files.index(file)] = merge(sspres[files.index(file)],
                                                  lsrow)
            ml = max_likelihood(np.array(mspr), np.array(aspr), np.ones(6) / 5)
            for i in range(6):
                ipmbias[i] = merge(ipmbias[i],
                                   pd.DataFrame(zip([f'sans_{sans}'], [ml[0]],
                                                    [ml[i + 1]]),
                                                columns=col[:-1]
                                                )
                                   )
            for file in files:
                sspres[files.index(file)].to_csv(sspdir +
                                                 f'/{snfl[files.index(file)]}')
            for i in range(6):
                ipmbias[i].to_csv(sspdir + f'/{mutations[i]}_panmut.csv')


# Generate color-coded bar graphs of spectra for all species in the data set:
def fig_bar_graphs(rdir, spd, data, fontsz, speclbl):
    titv = pd.read_csv(rdir + '/by_mutation/ti_bias_mutation_vs_outcomes.csv')
    spti = list(titv.sort_values(by='mu_bias')['species'])
    if len(spti) > 10:
        for x in range(10, len(spti)):
            fontsz -= 1
    sptx = [sp[0].upper() + '. ' + sp[2:] for sp in spti]
    axlbl = {'paths': 'Fraction available missense mutations',
             'genome_mu': 'Fraction ' + r'$de$ ' + r'$novo$ ' +
                          'mutations (overall)',
             'mu': 'Fraction ' + r'$de$ ' + r'$novo$ ' +
                   'mutations (missense)',
             'adaptive': 'Fraction adaptive changes'}
    spml = [list(pd.read_csv(spd +
                             f'/{sp}_outcomes_model_vs_empirical_{data}.csv')
                 [speclbl]) for sp in spti
            ]
    spm = np.array(spml).T
    fig, ax = plt.subplots(figsize=(10, 10))
    btms, xtxt = np.zeros(len(spti)), -0.3
    for mi in ['A>G', 'G>A', 'A>C', 'A>T', 'G>C', 'G>T']:
        ytxt = 1.04
        if mi in ['A>G', 'A>C', 'G>C']:
            ytxt = 1.12
            if mi != 'A>G':
                xtxt += 4.8
        ax.text(xtxt, ytxt, f'{mi[0]}:{bp[mi[0]]}>{mi[-1]}:{bp[mi[-1]]}',
                fontsize=32, weight='bold', color=colors[mi],
                horizontalalignment='left'
                )
        ax.bar(sptx, spm[mutations.index(mi)], bottom=btms,
               color=colors[mi], alpha=2 / 3)
        plt.xticks(rotation=45, ha='right', fontsize=fontsz,
                   fontstyle='italic')
        plt.yticks(fontsize=24)
        btms += spm[mutations.index(mi)]
    plt.margins(y=0)
    ax.add_artist(lines.Line2D([0, 4], [1.1, 1.1], color='black', linewidth=3))
    ax.set_ylabel(axlbl[speclbl], fontsize=32, labelpad=20)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(which='both', width=2, length=8, right=True)
    plt.savefig(rdir + f'/figures/{speclbl}_spectra.pdf', bbox_inches='tight')
    plt.close('all')


# Graph regressions on adaptive vs mutation bias across the six
# single-nucleotide mutation classes, on a per-species basis:
def fig_intra_species_regr(data, andir, spdir, fdir):
    if not os.path.exists(fdir + '/within_species'):
        os.makedirs(fdir + '/within_species')
    intracol = ["species", "beta", "null beta", "Pearson's r", "p"]
    tbl = pd.DataFrame(columns=intracol)
    for sp in list(sp_mutn['species']):
        spfile = f'{sp}_intra_sp_bias_adaptive.csv'
        biases = pd.read_csv(spdir +
                             f'/{sp}_outcomes_model_vs_empirical_{data}.csv')
        regr = pd.read_csv(andir + f'/model/bootstrap_results/adaptive'
                                   f'/within_species/{spfile}')
        nrgr = pd.read_csv(andir + f'/null/bootstrap_results/adaptive'
                                   f'/within_species/{spfile}')
        rsts = pd.read_csv(andir + f'/model/statistics/adaptive'
                                   f'/within_species/{spfile}')
        nsts = pd.read_csv(andir + f'/null/statistics/adaptive'
                                   f'/within_species/{spfile}')
        si = list(rsts['var']).index('slope')
        ni = list(nsts['var']).index('slope')
        ri = list(rsts['var']).index('r')
        slp = num_form([list(rsts['val'])[si], list(rsts['-95%'])[si],
                        list(rsts['+95%'])[si]])
        nlp = num_form([list(nsts['val'])[ni], list(nsts['-95%'])[ni],
                        list(nsts['+95%'])[ni]])
        cor = num_form([list(rsts['val'])[ri], list(rsts['-95%'])[ri],
                        list(rsts['+95%'])[ri]])

        nulb = list(nrgr['slope'])[1:]
        rgrb = float(list(regr['slope'])[0])
        pval = rsig_simdat(rgrb, nulb)
        if pval[0] == '<':
            ptxt = f"p{pval}"
        else:
            ptxt = f"p={pval}"
        tbl = merge(tbl,
                    pd.DataFrame(zip([species_name(sp)],
                                     [f'{slp[0]} ({slp[1]} \u2014 {slp[2]})'],
                                     [f'{nlp[0]} ({nlp[1]} \u2014 {nlp[2]})'],
                                     [f'{cor[0]} ({cor[1]} \u2014 {cor[2]})'],
                                     [ptxt]), columns=intracol
                                 )
                    )
        slps, itc = np.array(regr['slope']), np.array(regr['intercept'])
        mb = np.array(list(biases['mu_bias']))
        ab = np.array(list(biases['a_bias']))
        axv = axis_vals(min(np.concatenate((mb, ab))),
                        max(np.concatenate((mb, ab))), True)
        logx = np.log10(axv[1][-1]) - np.log10(axv[1][0])
        xmsr = axmeasures(axv)
        txvrt = 10 ** (xmsr[1] - (xmsr[2] * 0.075))
        txhrz = 10 ** (xmsr[0] + (xmsr[2] * 0.025))
        fig, ax = plt.subplots(figsize=(10, 10))
        # OPTIONAL: UNCOMMENT TO PLOT ALL BOOTSTRAP REGRESSION LINES
        # for bi in range(1, sim_n):
        #     ax.plot(axv[1], 10 ** ((np.log10(axv[1]) * slps[bi]) + itc[bi]),
        #             linewidth=0.5, color='xkcd:silver', alpha=0.5)
        ax.plot(axv[1], 10 ** ((np.log10(axv[1]) * slps[0]) + itc[0]),
                linewidth=2, color='black', zorder=5)
        for mi in range(len(mb)):
            ax.text(mb[mi], ab[mi], ucdsps[mi], color=colors[mutations[mi]],
                    fontsize=48, horizontalalignment='center',
                    verticalalignment='center', alpha=2 / 3
                    )
        ax.text(txhrz, txvrt, species_name(sp), fontsize=44,
                fontstyle='italic', zorder=10)
        ax.text(10 ** (np.log10(axv[1][-1]) - (logx / 50)),
                10 ** (np.log10(axv[1][0]) + (logx / 50)), ptxt,
                color='black', fontsize=40, verticalalignment='bottom',
                horizontalalignment='right', zorder=15
                )
        ax.text(10 ** (np.log10(axv[1][-1]) - (logx / 50)),
                10 ** (np.log10(axv[1][0]) + (logx / 10)),
                f"$\u03B2$={slp[0]}",
                color='black', fontsize=40, verticalalignment='bottom',
                horizontalalignment='right', zorder=15
                )
        # OPTIONAL: INDIVIDUAL AXIS LABELS FOR EACH PLOT
        # (DEFAULT: Collective labels to be added manually)
        # ax.set_xlabel('Mutation bias', fontsize=50)
        # ax.set_ylabel('Bias in adaptive outcome', fontsize=50)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(np.array(axv[0]), axv[0], rotation=axv[2], fontsize=40)
        ax.set_yticks(np.array(axv[0]), axv[0], fontsize=40)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.tick_params(which='both', width=2, top=True, right=True)
        ax.tick_params(which='minor', length=7.5, top=True, right=True)
        ax.tick_params(which='major', length=15, top=True, right=True)
        ax.set_xbound(lower=np.array(axv[1])[0],
                      upper=np.array(axv[1])[-1])
        ax.set_ybound(lower=np.array(axv[1])[0],
                      upper=np.array(axv[1])[-1])
        plt.savefig(fdir + f'/within_species/{sp}_mu_vs_adp.pdf',
                    bbox_inches='tight')
        plt.close('all')
    tbl.to_csv(andir + f'/model/statistics/adaptive/within_species'
                       f'/intra_species_regressions_stats.csv')


# Graph regression that aggregates intra- & interspecies variation in bias:
def fig_inter_x_intra_species(rdir, mdl, btyp, spec):
    sdir = rdir + f'/analysis/{mdl}/statistics'
    bdir = rdir + f'/analysis/{mdl}/bootstrap_results/{spec}/general_bias'
    regr = pd.read_csv(bdir + f'/inter_x_intra_species_bias_{spec}.csv')
    regs = pd.read_csv(sdir + f'/{spec}/general_bias'
                              f'/inter_x_intra_species_bias_{spec}.csv')
    msts = pd.DataFrame(columns=['var', 'val', 'mid', 'err', '+95%', '-95%'])
    asts = pd.DataFrame(columns=['var', 'val', 'mid', 'err', '+95%', '-95%'])
    ssl = regs.loc[regs['var'] == 'slope']
    sr = num_form([round(ssl[l].iloc[0], 2)
                   for l in ['-95%', '+95%']])
    if abs(float(sr[0])) < 0.001:
        lo = '0.0'
    else:
        lo = sr[0]
    txt = f"Effect ($\u03B2$): " \
          f"{round(ssl['val'].iloc[0], 2)} ({lo} - {sr[1]})"
    for m in mutations:
        mst = pd.read_csv(sdir + f'/de_novo/general_bias'
                                 f'/{m}_general_bias_de_novo.csv')
        ast = pd.read_csv(sdir + f'/adaptive/general_bias'
                                 f'/{m}_general_bias_adaptive.csv')
        msts, asts = merge(msts, mst[:-3]), merge(asts, ast[:-3])
    mdpt, adpt = np.array(msts['val']), np.array(asts['val'])
    merr, aerr = np.array(msts['err']), np.array(asts['err'])
    mmid, amid = np.array(msts['mid']), np.array(asts['mid'])
    axv = axis_vals(min(np.array(list(msts['-95%']) + list(asts['-95%']))),
                    max(np.array(list(msts['+95%']) + list(asts['+95%']))),
                    True
                    )
    logx = (np.log10(axv[1][-1]) - np.log10(axv[1][0]))
    fig, ax = plt.subplots(figsize=(10, 10))
    slp, itc = np.array(regr['slope']), np.array(regr['intercept'])
    for bi in range(1, len(slp)):
        a_locr = 10 ** ((np.log10(axv[1]) * slp[bi]) + itc[bi])
        ax.plot(axv[1], a_locr, linewidth=0.5, color='xkcd:silver',
                alpha=0.5)
    ax.plot(axv[1], 10 ** ((np.log10(axv[1]) * slp[0]) + itc[0]),
            linewidth=2, color='black', zorder=10)
    ax.scatter(mdpt, adpt, s=100, color='black', zorder=15)
    ax.errorbar(mmid, adpt, xerr=merr, fmt='none', linewidth=2,
                color='black', zorder=15)
    ax.errorbar(mdpt, amid, yerr=aerr, fmt='none', linewidth=2,
                color='black', zorder=15)
    ax.set_xlabel('Mutation bias', fontsize=40)
    ax.set_ylabel('Bias in adaptive outcome', fontsize=40)
    ax.text(10 ** (np.log10(axv[1][-1]) - (logx / 50)),
            10 ** (np.log10(axv[1][0]) + (logx / 50)), txt,
            color='black', fontsize=36, verticalalignment='bottom',
            horizontalalignment='right'
            )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(np.array(axv[0]), axv[0], rotation=axv[2], fontsize=36)
    ax.set_yticks(np.array(axv[0]), axv[0], fontsize=36)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(which='both', width=2, top=True, right=True)
    ax.tick_params(which='minor', length=7.5, top=True, right=True)
    ax.tick_params(which='major', length=15, top=True, right=True)
    ax.set_xbound(lower=np.array(axv[1])[0], upper=np.array(axv[1])[-1])
    ax.set_ybound(lower=np.array(axv[1])[0], upper=np.array(axv[1])[-1])
    plt.savefig(rdir + f'/figures/{mdl}'
                       f'/inter_x_intra_species_bias_{spec}.pdf',
                bbox_inches='tight'
                )
    plt.close('all')
    print(f"{mdl}-result regressions plotted for "
          f"{btyp}-biased {spec} spectrum")


# Graph individual regressions on interspecies variation:
def fig_xspecies_regressions(rdir, mdl, spec):
    xf, ff, rf = 'analysis', 'figures', 'bootstrap_results'
    mf, af, sf = 'de_novo', 'adaptive', 'statistics'
    for pth, dirs, files in os.walk(rdir, topdown=True):
        if len(dirs) == 0:
            speci = pth.index(spec)
            brp = pth[:pth.index(sf)] + rf + pth[pth.index(sf) + len(sf):]
            mp = pth[:speci] + mf + pth[speci + len(spec):]
            ap = pth[:speci] + af + pth[speci + len(spec):]
            fp = pth[:pth.index(xf)] + f'{ff}/{mdl}' + pth[speci + len(spec):]
            if not os.path.exists(fp):
                os.makedirs(fp)
            for file in files:
                if 'panmut' not in file and 'intra_species' not in file:
                    btyp, sclr = file[:file.index('_bias_')], 'black'
                    rgst = pd.read_csv(pth + f'/{file}')
                    slp = rgst.loc[rgst['var'] == 'slope']
                    sr = num_form([round(slp[l].iloc[0], 2)
                                   for l in ['-95%', '+95%']])
                    if abs(float(sr[0])) < 0.001:
                        lo = '0.0'
                    else:
                        lo = sr[0]
                    txt = f"Effect ($\u03B2$): " \
                          f"{round(slp['val'].iloc[0], 2)} ({lo} - {sr[1]})"
                    mus = pd.read_csv(mp + f'/{btyp}_bias_{mf}.csv')[:-3]
                    ads = pd.read_csv(ap + f'/{btyp}_bias_{af}.csv')[:-3]
                    axv = axis_vals(min(np.array(list(mus['-95%']) +
                                                 list(ads['-95%']))),
                                    max(np.array(list(mus['+95%']) +
                                                 list(ads['+95%']))), False
                                    )
                    logx = np.log10(axv[1][-1]) - np.log10(axv[1][0])
                    xmsr = axmeasures(axv)
                    axmin, axmax, axrng = xmsr[0], xmsr[1], xmsr[2]
                    txvrt = 10 ** (axmax - (axrng * 0.075))
                    txhrz = 10 ** (axmin + (axrng * 0.025))
                    mdpt, adpt = np.array(mus['val']), np.array(ads['val'])
                    mmid, amid = np.array(mus['mid']), np.array(ads['mid'])
                    merr, aerr = np.array(mus['err']), np.array(ads['err'])
                    fig, ax = plt.subplots(figsize=(10, 10))
                    slp = np.array(pd.read_csv(brp + f'/{file}')['slope'])
                    itc = np.array(pd.read_csv(brp + f'/{file}')['intercept'])
                    if '>' in btyp:
                        refb, altb = btyp[0], btyp[2]
                        sclr = colors[f'{refb}>{altb}']
                        ax.text(txhrz, txvrt,
                                f'{refb}:{bp[refb]}>{altb}:{bp[altb]}',
                                color=colors[f'{refb}>{altb}'],
                                weight='bold', fontsize=40)
                    for bi in range(1, len(slp)):
                        a_locr = 10 ** ((np.log10(axv[1]) * slp[bi]) + itc[bi])
                        ax.plot(axv[1], a_locr, linewidth=0.5,
                                color='xkcd:silver', alpha=0.5)
                    ax.plot(axv[1],
                            10 ** ((np.log10(axv[1]) * slp[0]) + itc[0]),
                            linewidth=2, color='black', zorder=5
                            )
                    mpos = [10 ** ((logx / mx) + np.log10(axv[1][-1]))
                            for mx in [16, 10]]
                    for si in range(len(mus)):
                        ax.errorbar(mmid[si], adpt[si], xerr=merr[si],
                                    capsize=8, capthick=2, linewidth=2,
                                    color=sclr, alpha=1 / 3, zorder=10
                                    )
                        ax.errorbar(mdpt[si], amid[si], yerr=aerr[si],
                                    capsize=8, capthick=2, linewidth=2,
                                    color=sclr, alpha=1 / 3, zorder=10
                                    )
                        if len(mus) <= 20:
                            ax.text(mdpt[si], adpt[si], ucdsym[si], color=sclr,
                                    fontsize=28, horizontalalignment='center',
                                    verticalalignment='center', zorder=15
                                    )
                            apos = [10 ** (np.log10(axv[1][-1]) -
                                           (a / 200) - (si * logx / 20))
                                    for a in [0, logx]
                                    ]
                            ax.text(mpos[0], apos[0], ucdsym[si], color=sclr,
                                    fontsize=24, horizontalalignment='center',
                                    verticalalignment='top'
                                    )
                            ax.text(mpos[1], apos[1],
                                    species_name(mus['var'][si]), fontsize=24,
                                    fontstyle='italic',
                                    horizontalalignment='left',
                                    verticalalignment='top'
                                    )
                        else:
                            ax.scatter(mdpt[si], adpt[si], s=100,
                                       color=sclr, alpha=1 / 3, zorder=15)
                    ax.set_xlabel('Mutation bias', fontsize=40)
                    ax.set_ylabel('Bias in adaptive outcome', fontsize=40)
                    ax.text(10 ** (np.log10(axv[1][-1]) - (logx / 50)),
                            10 ** (np.log10(axv[1][0]) + (logx / 50)), txt,
                            color='black', fontsize=36,
                            verticalalignment='bottom',
                            horizontalalignment='right'
                            )
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.set_xticks(np.array(axv[0]), axv[0], rotation=axv[2],
                                  fontsize=36)
                    ax.set_yticks(np.array(axv[0]), axv[0], fontsize=36)
                    for spine in ax.spines.values():
                        spine.set_linewidth(2)
                    ax.tick_params(which='both', width=2, top=True, right=True)
                    ax.tick_params(which='minor', length=7.5, top=True,
                                   right=True)
                    ax.tick_params(which='major', length=15, top=True,
                                   right=True)
                    ax.set_xbound(lower=np.array(axv[1])[0],
                                  upper=np.array(axv[1])[-1])
                    ax.set_ybound(lower=np.array(axv[1])[0],
                                  upper=np.array(axv[1])[-1])
                    plt.savefig(fp + f'/{file[:-4]}.pdf', bbox_inches='tight')
                    plt.close('all')


# Graph regressions that isolate variation across species in mutation bias
# that aggregates the results from all six mutation classes:
def fig_multi_regressions(rdir, sp_n, mdl, btyp, spec):
    fdir = rdir[:rdir.index('analysis')] + f'figures/{mdl}/{btyp}_bias'
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    sdir = rdir + f'/{mdl}/statistics'
    bdir = rdir + f'/{mdl}/bootstrap_results/{spec}'
    msts = pd.DataFrame(columns=['var', 'val', 'mid', 'err', '+95%', '-95%'])
    asts = pd.DataFrame(columns=['var', 'val', 'mid', 'err', '+95%', '-95%'])
    for m in mutations:
        mst = pd.read_csv(sdir + f'/de_novo/{btyp}_bias'
                                 f'/{m}_intersp_panmut_{btyp}_bias_de_novo'
                                 f'.csv'
                          )
        ast = pd.read_csv(sdir + f'/adaptive/{btyp}_bias'
                                 f'/{m}_intersp_panmut_{btyp}_bias_adaptive'
                                 f'.csv'
                          )
        msts, asts = merge(msts, mst[:-2]), merge(asts, ast[:-2])
    mdpt, adpt = np.array(msts['val']), np.array(asts['val'])
    merr, aerr = np.array(msts['err']), np.array(asts['err'])
    mmid, amid = np.array(msts['mid']), np.array(asts['mid'])
    axv = axis_vals(min(np.array(list(msts['-95%']) + list(asts['-95%']))),
                    max(np.array(list(msts['+95%']) + list(asts['+95%']))),
                    True
                    )
    logx = np.log10(axv[1][-1]) - np.log10(axv[1][0])
    xmsr = axmeasures(axv)
    axmin, axmax, axrng = xmsr[0], xmsr[1], xmsr[2]
    txvrt = 10 ** (axmax - (axrng * 0.075))
    txhrz = 10 ** (axmin + (axrng * 0.025))
    mpos = [10 ** ((axrng / mx) + np.log10(axv[1][-1])) for mx in [16, 10]]
    for m in mutations:
        mdx = int(round(mutations.index(m) * sp_n))
        mlbl = f'{m[0]}:{bp[m[0]]}>{m[-1]}:{bp[m[-1]]}'
        mdi = pd.read_csv(sdir + f'/de_novo/{btyp}_bias'
                                 f'/{m}_intersp_panmut_{btyp}_bias_de_novo'
                                 f'.csv'
                          )
        adi = pd.read_csv(sdir + f'/adaptive/{btyp}_bias'
                                 f'/{m}_intersp_panmut_{btyp}_bias_adaptive'
                                 f'.csv'
                          )
        alog = pd.read_csv(sdir + f'/adaptive/{btyp}_bias'
                                  f'/{m}_intersp_panmut_{btyp}_bias_adaptive'
                                  f'.csv'
                           )
        ssl = alog.loc[alog['var'] == 'slope']
        sr = num_form([round(ssl[l].iloc[0], 2) for l in ['-95%', '+95%']])
        if abs(float(sr[0])) < 0.001:
            lo = '0.0'
        else:
            lo = sr[0]
        txt = f"Effect ($\u03B2$): " \
              f"{round(ssl['val'].iloc[0], 2)} ({lo} - {sr[1]})"
        mdv, adv = np.array(mdi['val'][:-2]), np.array(adi['val'][:-2])
        fig, ax = plt.subplots(figsize=(10, 10))
        rgs = pd.read_csv(bdir + f'/{btyp}_bias'
                                 f'/{m}_intersp_panmut_{btyp}_bias_{spec}.csv')
        slp = np.array(rgs['slope'])
        itc = np.array(rgs['intercept'])
        for bi in range(1, len(slp)):
            a_locr = 10 ** ((np.log10(axv[1]) * slp[bi]) + itc[bi])
            ax.plot(axv[1], a_locr, linewidth=0.5, color='xkcd:silver',
                    alpha=0.5)
        ax.errorbar(mmid, adpt, xerr=merr, fmt='none',
                    linewidth=2, color='xkcd:light grey', zorder=5)
        ax.errorbar(mdpt, amid, yerr=aerr, fmt='none',
                    linewidth=2, color='xkcd:light grey', zorder=5)
        ax.scatter(mdpt, adpt, s=100, color='xkcd:light grey', zorder=5)
        ax.plot(axv[1], 10 ** ((np.log10(axv[1]) * slp[0]) + itc[0]),
                linewidth=2, color='black', zorder=10)
        ax.errorbar(mmid[mdx:mdx + sp_n], adpt[mdx:mdx + sp_n],
                    xerr=merr[mdx:mdx + sp_n], fmt='none', linewidth=2,
                    capsize=8, capthick=2, color=colors[m], alpha=0.5,
                    zorder=15,
                    )
        ax.errorbar(mdpt[mdx:mdx + sp_n], amid[mdx:mdx + sp_n],
                    yerr=aerr[mdx:mdx + sp_n], fmt='none', linewidth=2,
                    capsize=8, capthick=2, color=colors[m], alpha=0.5,
                    zorder=15
                    )
        for si in range(len(mdv)):
            if len(mdv) <= 20:
                sp = species_name(mdi['var'][si])
                ax.text(mdv[si], adv[si], ucdsym[si], color=colors[m],
                        fontsize=28, horizontalalignment='center',
                        verticalalignment='center', zorder=15
                        )
                apos = [10 ** (np.log10(axv[1][-1]) -
                               (a / 200) - (si * axrng / 20))
                        for a in [0, axrng]
                        ]
                ax.text(mpos[0], apos[0], ucdsym[si], color=colors[m],
                        fontsize=24, horizontalalignment='center',
                        verticalalignment='top'
                        )
                ax.text(mpos[1], apos[1], sp, fontsize=24,
                        fontstyle='italic', horizontalalignment='left',
                        verticalalignment='top'
                        )
            else:
                ax.scatter(mdv, adv, s=160, color=colors[m],
                           edgecolors='black', zorder=20)
        ax.text(10 ** (np.log10(axv[1][-1]) - (logx / 50)),
                10 ** (np.log10(axv[1][0]) + (logx / 50)), txt,
                color='black', fontsize=36,
                verticalalignment='bottom',
                horizontalalignment='right', zorder=20
                )
        ax.set_xlabel('Mutation bias', fontsize=40)
        ax.set_ylabel('Bias in adaptive outcome', fontsize=40)
        ax.text(txhrz, txvrt, mlbl, color=colors[m], weight='bold',
                fontsize=40)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(np.array(axv[0]), axv[0], rotation=axv[2], fontsize=36)
        ax.set_yticks(np.array(axv[0]), axv[0], fontsize=36)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.tick_params(which='both', width=2, top=True, right=True)
        ax.tick_params(which='minor', length=7.5, top=True, right=True)
        ax.tick_params(which='major', length=15, top=True, right=True)
        ax.set_xbound(lower=np.array(axv[1])[0], upper=np.array(axv[1])[-1])
        ax.set_ybound(lower=np.array(axv[1])[0], upper=np.array(axv[1])[-1])
        plt.savefig(fdir + f'/{mdl}_{m}_{spec}_{btyp}_bias.pdf',
                    bbox_inches='tight')
        plt.close('all')
    print(f"{mdl}-result regressions plotted for "
          f"{btyp}-biased {spec} spectrum")


# Graph the regressions resulting from leaving each species out one at a time:
def fig_sans_species(rdir, sspdir, fdir):
    if not os.path.exists(fdir + '/sans_species'):
        os.makedirs(fdir + '/sans_species')
    for root, dirs, files in os.walk(sspdir, topdown=True):
        for file in files:
            regrs = pd.read_csv(sspdir + f'/{file}')
            scol = [col for col in regrs.columns if 'slope' in col]
            icol = [col for col in regrs.columns if 'intercept' in col]
            slopes, intercepts = regrs[scol[0]], regrs[icol[0]]
            sdir = rdir + f'/analysis/model/statistics'
            sts = pd.DataFrame(columns=['var', 'val', 'mid',
                                        'err', '+95%', '-95%'])
            if file[:3] in mutations:
                m = file[:3]
                rclr = colors[file[:3]]
                if 'panmut' in file:
                    res = pd.read_csv(sdir +
                                      f'/adaptive/general_bias'
                                      f'/{m}'
                                      f'_intersp_panmut_general_bias_adaptive'
                                      f'.csv'
                                      )
                else:
                    res = pd.read_csv(sdir + f'/adaptive/general_bias'
                                             f'/{m}_general_bias_adaptive.csv')
            else:
                rclr = 'grey'
                if 'transition' in file:
                    res = pd.read_csv(sdir +
                                      f'/adaptive/transition_bias/{file}')
                else:
                    res = pd.read_csv(sdir +
                                      f'/adaptive/general_bias'
                                      f'/inter_x_intra_species_bias_adaptive'
                                      f'.csv'
                                      )
            if 'transition' in file:
                mst = pd.read_csv(sdir + f'/de_novo/transition_bias'
                                         f'/transition_bias_de_novo.csv')
                ast = pd.read_csv(sdir + f'/adaptive/transition_bias'
                                         f'/transition_bias_adaptive.csv')
                sts = merge(sts, mst[:-3])
                sts = merge(sts, ast[:-3])
            else:
                for m in mutations:
                    mst = pd.read_csv(sdir + f'/de_novo/general_bias'
                                             f'/{m}_general_bias_de_novo.csv')
                    ast = pd.read_csv(sdir + f'/adaptive/general_bias'
                                             f'/{m}_general_bias_adaptive.csv')
                    sts = merge(sts, mst[:-3])
                    sts = merge(sts, ast[:-3])
            axv = axis_vals(min(np.array(list(sts['-95%']) +
                                         list(sts['-95%']))),
                            max(np.array(list(sts['+95%']) +
                                         list(sts['+95%']))),
                            True)
            xmsr = axmeasures(axv)
            txvrt = 10 ** (xmsr[1] - (xmsr[2] * 0.075))
            txhrz = 10 ** (xmsr[0] + (xmsr[2] * 0.025))
            fig, ax = plt.subplots(figsize=(10, 10))
            slps = num_form([min(list(slopes)), max(list(slopes))])
            if abs(float(slps[0])) < 0.001:
                lo = '0.0'
            else:
                lo = slps[0]
            ax.text(10 ** (xmsr[1] - (xmsr[2] * 0.025)), txhrz,
                    f"Effect ($\u03B2$): {lo} - {slps[1]}",
                    fontsize=36, horizontalalignment='right'
                    )
            for i in range(len(slopes)):
                reg_i = 10 ** ((np.log10(axv[1]) * slopes[i]) + intercepts[i])
                ax.plot(axv[1], reg_i, linewidth=4, color=rclr, alpha=1 / 4)
            slpi = [i for i in range(len(res))
                    if 'slope' in list(res['var'])[i]][0]
            icpi = [i for i in range(len(res))
                    if 'intercept' in list(res['var'])[i]][0]
            slp, itcp = list(res['val'])[slpi], list(res['val'])[icpi]
            ax.plot(axv[1], 10 ** ((np.log10(axv[1]) * slp) + itcp),
                    linewidth=3, color='black', zorder=5)
            if file[:3] in mutations:
                ax.text(txhrz, txvrt,
                        f"{file[0]}:{bp[file[0]]}>{file[2]}:{bp[file[2]]}",
                        color=rclr, weight='bold', fontsize=40, zorder=10
                        )
            ax.set_xlabel('Mutation bias', fontsize=40)
            ax.set_ylabel('Bias in adaptive outcome', fontsize=40)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xticks(np.array(axv[0]), axv[0], rotation=axv[2],
                          fontsize=36)
            ax.set_yticks(np.array(axv[0]), axv[0], fontsize=36)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            ax.tick_params(which='both', width=2, top=True, right=True)
            ax.tick_params(which='minor', length=7.5, top=True, right=True)
            ax.tick_params(which='major', length=15, top=True, right=True)
            ax.set_xbound(lower=np.array(axv[1])[0],
                          upper=np.array(axv[1])[-1])
            ax.set_ybound(lower=np.array(axv[1])[0],
                          upper=np.array(axv[1])[-1])
            plt.savefig(fdir + f'/sans_species/{file[:-4]}.pdf',
                        bbox_inches='tight')
            plt.close('all')


# Graph the summarized results, slope (beta), for the regressions that result
# from leaving each species out one at a time:
def fig_sans_species_effects(ssd, ssfd, col):
    fx, sp = pd.DataFrame(columns=col), list(sp_mutn['species'])
    ti = pd.read_csv(ssd + '/transition_bias_adaptive.csv'
                     ).sort_values(by='slope', ignore_index=True)['slope']
    ii = pd.read_csv(ssd + '/all_snm_bias_adaptive.csv'
                     ).sort_values(by='slope', ignore_index=True)['slope']
    er = pd.read_csv(ssd + '/A>C_panmut'
                           '.csv').sort_values(by='slope',
                                               ignore_index=True)['slope']
    bias = [['Transition' for s in range(len(sp))],
            ['Intra- & interspecies' for s in range(len(sp))],
            ['Interspecies' for s in range(len(sp))]
            ]
    blk = ['black' for s in range(len(sp))]
    fx = merge(fx, pd.DataFrame(zip(bias[0], list(ti), blk), columns=col))
    fx = merge(fx, pd.DataFrame(zip(bias[1], list(ii), blk), columns=col))
    fx = merge(fx, pd.DataFrame(zip(bias[2], list(er), blk), columns=col))
    clrs = ['black', 'black', 'black'] + [colors[m] for m in mutations]
    for m in mutations:
        ml = [f'{m[0]}:{bp[m[0]]}>{m[2]}:{bp[m[2]]}'
              for s in range(len(sp))]
        mi = pd.read_csv(ssd + f'/{m}_bias_adaptive.csv'
                         ).sort_values(by='slope', ignore_index=True)['slope']
        clr = [colors[m] for s in range(len(sp))]
        fx = merge(fx, pd.DataFrame(zip(ml, list(mi), clr), columns=col))
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.stripplot(data=fx, x="Regression", y="Effect ($\u03B2$)", size=10,
                  hue="Regression", palette=clrs, alpha=0.5, ax=ax)
    ax.plot(np.array([0, 8]), np.zeros(2), linewidth=1.5,
            linestyle=':', color='black')
    ax.set_ybound(lower=-0.5, upper=1.5)
    ax.set_yticks([-0.5, 0, 0.5, 1.0, 1.5], ['-0.5', '0', '0.5', '1', '1.5'],
                  fontsize=14)
    ax.set_xlabel(xlabel="Regression", fontsize=16)
    ax.set_ylabel(ylabel="Effect ($\u03B2$)", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(ssfd + f'/sans_species/effects_per_regression.pdf',
                bbox_inches='tight')
    plt.close('all')


# Generate output figures of spectra and regressions on the
# mutation and adaptive data:
def figures(rdir, data, models, spectra):
    if data == 'all_data':
        models += ['null']
    if not os.path.exists(rdir + '/figures'):
        os.makedirs(rdir + '/figures')
    fig_sans_species(rdir, rdir + f'/sans_one_species', rdir + '/figures')
    fig_sans_species_effects(rdir + '/sans_one_species', rdir + '/figures',
                             ['Regression', 'Effect ($\u03B2$)', 'color'])
    fig_intra_species_regr(data, rdir + '/analysis',
                           rdir + '/by_species', rdir + '/figures')
    for spectrum in spectra:
        if spectrum == 'de_novo':
            for speclabel in ['paths', 'genome_mu', 'mu']:
                fig_bar_graphs(rdir, rdir + '/by_species', data, 28, speclabel)
        else:
            fig_bar_graphs(rdir, rdir + '/by_species', data, 28, 'adaptive')
        for model in models:
            fig_xspecies_regressions(rdir + f'/analysis/{model}/statistics'
                                            f'/{spectrum}', model, spectrum)
            for biastype in btypes[2:]:
                fig_multi_regressions(rdir + '/analysis',
                                      len(list(sp_mutn['species'])), model,
                                      biastype, spectrum
                                      )
                fig_inter_x_intra_species(rdir, model, biastype, spectrum)


# Generate graph summarizing the results from assessing the impact of
# neutral contamination of adaptive data on the mutation-vs-adaptation
# regressions:
def fig_contam_effect(ac):
    bias = ['transition_bias', 'inter_x_intra_species_bias',
            f'{mutations[0]}_intersp_panmut_general_bias']
    bdir = ['transition', 'general', 'general']
    lbls = ['Effect (transition)', 'Effect (intra-/inter-)',
            'Effect (interspecies)']
    cols = ['effect', 'errmid', 'err', 'p']
    sttl = [pd.DataFrame(columns=cols), pd.DataFrame(columns=cols),
            pd.DataFrame(columns=cols)]
    stmi = [pd.DataFrame(columns=cols) for m in mutations]
    trmd = [s for s in os.listdir(f'./results'
                                  f'/results_with_stats_on_{sim_n}_sims')
            if 'trimmed_data' in s]
    trmd.sort()
    for subd in ['all_data'] + trmd:
        sdir = f'./results/results_with_stats_on_{sim_n}_sims/{subd}' \
               f'/analysis/model/statistics/adaptive'
        nuld = f'./results/results_with_stats_on_{sim_n}_sims' \
               f'/{subd}/analysis/null/bootstrap_results/adaptive'
        for bi in bias:
            iso = pd.read_csv(sdir + f'/{bdir[bias.index(bi)]}_bias'
                                     f'/{bi}_adaptive.csv')
            nul = np.array(list(pd.read_csv(nuld +
                                            f'/{bdir[bias.index(bi)]}_bias'
                                            f'/{bi}_adaptive.csv'
                                            )['slope']
                                )
                           )[1:]
            sttl[bias.index(bi)] = sumstats(iso, nul, 'slope',
                                            sttl[bias.index(bi)], cols)
        for m in mutations:
            iso = pd.read_csv(sdir + f'/general_bias/'
                                     f'/{m}_general_bias_adaptive.csv')
            nul = np.array(list(pd.read_csv(nuld +
                                            f'/general_bias/'
                                            f'/{m}_general_bias_adaptive'
                                            f'.csv')['slope']
                                )
                           )[1:]
            stmi[mutations.index(m)] = sumstats(iso, nul, 'slope',
                                                stmi[mutations.index(m)], cols)
    fig, axs = plt.subplots(4, 2, figsize=(10, 10))
    cr = np.array([0] + ac)
    xspc = assumed_contam[-1] / 4
    mlbl = {'A>C': [xspc, 2.16], 'A>G': [0, 2.16], 'A>T': [xspc, 1.82],
            'G>A': [0, 1.82], 'G>C': [xspc * 2, 2.16], 'G>T': [xspc * 2, 1.82]}
    for r in range(0, 3):
        axs[r, 0].plot(cr, list(sttl[r]['effect']), linewidth=2, color='black')
        axs[r, 0].scatter(cr, list(sttl[r]['effect']), s=25, color='black')
        axs[r, 0].errorbar(cr, list(sttl[r]['errmid']),
                           yerr=list(sttl[r]['err']), fmt='none', capsize=2,
                           capthick=1, linewidth=1, color='black'
                           )
        axs[r, 0].set_yticks([-1, 0, 1, 2], ['-1', '0', '1', '2'], fontsize=12)
        axs[r, 0].set_ylabel(lbls[r], fontsize=14)
        axs[r, 1].plot(cr, list(sttl[r]['p']), linewidth=2, color='black')
        axs[r, 1].scatter(cr, list(sttl[r]['p']), s=25, color='black')
        axs[r, 1].set_ylabel(f'p ({lbls[r][8:-1]})', fontsize=14)
    for mi in range(0, 6):
        m = mutations[mi]
        axs[3, 0].plot(cr, list(stmi[mi]['effect']), linewidth=2,
                       color=colors[m], alpha=0.5)
        axs[3, 0].errorbar(cr, list(stmi[mi]['errmid']),
                           yerr=list(stmi[mi]['err']), fmt='none', capsize=2,
                           capthick=1, linewidth=1, color=colors[m], alpha=0.5
                           )
        axs[3, 0].text(mlbl[m][0], mlbl[m][1],
                       f'{m[0]}:{bp[m[0]]}>{m[2]}:{bp[m[2]]}',
                       color=colors[m], fontsize=12
                       )
        axs[3, 1].plot(cr, list(stmi[mi]['p']), linewidth=2,
                       color=colors[mutations[mi]], alpha=0.5)
    for r in range(0, 4):
        for rp in range(0, 2):
            axs[r, rp].set_xticks(cr, ['0', '0.1', '0.2', '0.3', '0.4'],
                                  fontsize=12)
        axs[r, 0].plot(cr, np.zeros(5), linewidth=1, linestyle=':',
                       color='black')
        axs[r, 1].plot(cr, np.ones(5) * 0.05, linewidth=1, linestyle=':',
                       color='black')
        axs[r, 1].text(0.4, 0.04, '(p=0.05)', horizontalalignment='right',
                       verticalalignment='top', fontsize=12)
        axs[r, 1].set_yscale('log')
        axs[r, 1].set_ybound(lower=0.00005, upper=1)
        axs[r, 1].set_yticks([0.0001, 0.001, 0.01, 0.1, 1],
                             ['0.0001', '0.001', '0.01', '0.1', '1'],
                             fontsize=12
                             )
        axs[r, 0].set_ybound(lower=-1, upper=2)
    axs[3, 0].set_ybound(lower=-1, upper=2.5)
    axs[3, 0].set_yticks([-1, 0, 1, 2], ['-1', '0', '1', '2'],
                         fontsize=12)
    axs[3, 0].set_ylabel('Effect (per SNM)', fontsize=14)
    axs[3, 1].set_ylabel('p (per SNM)', fontsize=14)
    for rp in range(0, 2):
        axs[3, rp].set_xlabel('Fraction attributed to contamination',
                              fontsize=14)
    plt.tight_layout()
    plt.savefig(f'./results/results_with_stats_on_{sim_n}_sims/'
                f'effects_non_adaptive_contamination.pdf')
    plt.close('all')


# Call the functions to analyze adaptive vs mutation data & graph the results:
for data in usedata:
    if data != 'all_data':
        cntm = assumed_contam[usedata.index(data) - 1]
    rdir = f'./results/results_with_stats_on_{sim_n}_sims/{data}'
    if not os.path.exists(rdir):
        os.makedirs(rdir)
        mva_compare(adir, rdir, data, genetic_code,
                    pd.DataFrame(mutations, columns=['mutation']))
    iter_sans_species(rdir)
    figures(rdir, data, ['model'], ['adaptive', 'de_novo'])
    summary_table(rdir + '/analysis', data)
fig_contam_effect(assumed_contam)