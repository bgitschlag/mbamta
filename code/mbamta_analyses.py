"""

Multi-taxon analysis of mutation-biased adaptation (mbamta)

This analysis script evaluates the putative effect of mutation bias
on adaptive substitutions, using empirical data, with statistical
analysis performed using a bootstrap approach and simulated data
under the null model in which mutation bias has no effect on
adaptive substitutions. See READ ME file for further
information and guidance.

"""


# Import packages that will be needed for this script:
import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize


# Create folder with specified subfolders, for saving output files:
def folder(folders, subfolders):

    for f in folders:
        [os.makedirs(f + s) for s in subfolders
         if not os.path.exists(f + s)]

    return folders


# Append a row to a DataFrame:
def merge(append_to, appendix):

    if append_to.empty:
        append_to = appendix
    else:
        append_to = pd.concat([append_to, appendix])

    return append_to


# Create species-specific dictionary of codon usage frequencies:
def codon_use(species):

    ctbl = pd.read_csv(f'./data/codon_use.csv')
    ctbl = ctbl.loc[ctbl['species'] == species, :]
    cfrq = {}

    for codon in range(len(ctbl)):
        cfrq[ctbl.iloc[codon]['codon']] = (ctbl.iloc[codon]['fraction'])

    return cfrq


# Trim an adaptive substitutions spectrum (emp) by removing
# a fraction with a neutral mutation spectrum (mspec)
def trim(cntm, mspec, emp):

    mspec = mspec / mspec.sum()

    def trimmed_error(x):
        est = (mspec * cntm) + (x * (1 - cntm))
        return sum((emp - est) ** 2)

    def residual(adaptive_spec):
        return adaptive_spec.sum() - 1

    m = minimize(
        trimmed_error,
        np.ones(6) / 6,
        method="SLSQP",
        bounds=np.array([[0, 1] for i in range(6)]),
        constraints={'type': 'eq', 'fun': residual}
    )

    if not m.success:
        raise RuntimeError(
            f'Adaptive-spectrum decontamination failed: {m.message}')

    aspec = np.array([m['x'][i] for i in range(6)])

    # Return normalized adaptive spectrum:
    return aspec / aspec.sum()


# Generate spectrum of available missense mutation paths,
# to be used as the null model for the adaptive substitutions
# spectrum and as the weights on the mutation spectrum to
# obtain the missense-specific mutation spectrum:
def path_spectrum(
        muts, u, code, bp,
        bs, codon_use, path_spec,
        spectrum, dnmx
):

    # Iterate across all codons:
    for codon in code:

        # Only use amino acid codons (exclude stop codons):
        if code[codon] != '*':

            pths_frm_cdn = np.zeros(6)
            cfreq = codon_use[codon]

            # Evaluate mutations for each codon position:
            for position in range(3):

                alt_bs = [n for n in bs if n != codon[position]]

                # Evaluate all alternative alleles
                # for each codon position:
                for alt_base in alt_bs:

                    mutation, cdn = '', list(codon)
                    cdn[position] = alt_base
                    mut = mutation.join(cdn)

                    # Make sure the mutation changes an amino acid
                    # but does not make a stop codon:
                    if code[mut] != '*' and code[codon] != code[mut]:

                        ref, alt, path = codon[position], alt_base, 1

                        # Confirm that reference codon is set to the
                        # purine component of the reference base-pair:
                        if codon[position] in [bs[1], bs[3]]:
                            ref, alt = bp[ref], bp[alt]

                        # If modeling evolutionary dynamics as a function
                        # of both mutation and amino acid-level effects,
                        # apply these effects as weights on paths:
                        if dnmx == 'mu_and_effects':
                            aa = f'{code[codon]}>{code[mut]}'
                            path *= u.loc[
                                u['aa'].str.contains(aa)
                            ]['u'].iloc[0]

                        # Codon contributions to nominal and
                        # weighted spectra of mutational paths:
                        pths_frm_cdn[muts.index(f'{ref}>{alt}')] += path
                        # counts[muts.index(f'{ref}>{alt}')] += 1

            # Take product of mutation paths from codon &
            # codon-use fraction & add to an array:
            path_spec += cfreq * pths_frm_cdn

    # Normalize and add as columns to table of path spectra:
    # spectrum['code_freq'] = counts / sum(counts)
    spectrum['paths'] = path_spec / path_spec.sum()

    return spectrum


# Specify the contribution, by species and mutation class,
# toward available mutational paths, de novo mutations,
# adaptive substitutions, and associated biases:
def m_id_cont(mutation_in, mutation_out):

    mutation_out[0] += mutation_in['paths']
    mutation_out[1] += mutation_in['mu']
    mutation_out[2] += mutation_in['adaptive']

    if len(mutation_out) == 7:

        mutation_out[3] += mutation_in['m_bias']
        mutation_out[4] += mutation_in['a_bias']
        mutation_out[5] += mutation_in['m_bcpath']
        mutation_out[6] += mutation_in['a_bcpath']

    return mutation_out


# Create list of numbers of adaptive substitution events
# per mutation class in a given species:
def adaptive_counter(
        species, mtns, bp, bs, cnts,
        correct_for_pseudorep, pseudorep
):

    data = pd.read_csv(f'./data/adaptive_changes.csv')
    data = data.loc[data['species'] == species, :]

    if correct_for_pseudorep:
        if species in list(pseudorep['species']):
            pseudo = pseudorep.loc[pseudorep['species'] == species]
            pseudo_cat = pseudo['query_cat'].iloc[0]
            pseudo_crit = pseudo['criteria'].iloc[0]
            data = data.loc[~data[pseudo_cat].isin(pseudo_crit), :]

    for m in range(len(data)):

        if data.iloc[m]['ref_base'] in [bs[0], bs[2]]:
            rn = data.iloc[m]['ref_base']
            an = data.iloc[m]['alt_base']

        else:
            rn = bp[data.iloc[m]['ref_base']]
            an = bp[data.iloc[m]['alt_base']]

        cnts[mtns.index(f"{rn}>{an}")] += int(data.iloc[m]['events'])

    # Return list of counts of adaptive events by mutation class:
    return cnts


# Calculate & save missense-specific mutation spectrum,
# adaptive substitutions spectrum, available mutational
# path spectrum, and associated biases:
def adaptations_per_species(
        sdir, measure, mutations,
        sp_mutn, cntm, s, d, bp, bs,
        gen_mu, adata, u, c, dnmx
):

    spm = pd.DataFrame(zip(mutations), columns=['mutation'])
    ps = path_spectrum(
        mutations, u, c, bp, bs, codon_use(s), np.zeros(6), spm, dnmx)

    m_n = int(
        sp_mutn.iloc[
            [list(sp_mutn['species']).index(s)]
        ]['mu_counts'].iloc[0]
    )

    # Create column of uniform spectrum, path densities array, and
    # rates-adjusted mutation counts (mu * N_mut, with pseudocount):
    spm['nominal'] = np.ones(6) / 6
    paths = np.array(ps['paths'])
    mu = ((np.array(gen_mu[s]) * m_n) + 1)

    # Adaptive spectra:
    a_n = np.array(adata)
    if d != 'all_data':
        a_n = trim(cntm, mu * paths, a_n / a_n.sum()) * (
                np.array(adata).sum() * (1 - cntm))
    spm['adaptive_n'] = a_n
    a = a_n + 1
    if measure == 'freqs':
        mu *= paths
    elif measure == 'rates':
        a = a / paths
    else:
        raise RuntimeError(
            "Mutation and adaptation spectra must be"
            "measured in 'rates' or frequencies ('freqs')"
        )

    # Mutation and adaptive spectra (freq or rates, plus biases):
    mu, a = mu / mu.sum(), a / a.sum()
    spm['mu'] = mu
    spm['adaptive'] = a
    spm['m_bias'] = mu / (1 - mu)
    spm['a_bias'] = a / (1 - a)

    bcps = [
        np.concatenate((
                x[:3] / (sum(x[:3]) - x[:3]),
                x[3:] / (sum(x[3:]) - x[3:])
            )) for x in [mu, a]
    ]

    spm['m_bcpath'] = np.array(bcps[0])
    spm['a_bcpath'] = np.array(bcps[1])

    spm.to_csv(
        f'{sdir}/{s}_outcomes_model_vs_empirical_{d}.csv',
        index=False
    )

    print(
        f"spectra of mutations and adaptive "
        f"changes for {s} saved to folder"
    )


# Save output files of spectra and bias values by mutation type:
def adp_per_mut_type(
        sdir, mdir, sp_mutn,
        data, col, template
):

    for m in range(len(template)):
        m_v_a = pd.DataFrame(columns=col)
        ref = template.iloc[m]['mutation'][0]
        alt = template.iloc[m]['mutation'][-1]

        for s in sp_mutn['species']:

            mvad = pd.read_csv(
                f'{sdir}'
                f'/{s}_outcomes_model_vs_empirical_{data}.csv'
            )

            for n in range(len(mvad)):

                mi = mvad.iloc[n]

                if mi['mutation'][0] == ref and mi['mutation'][-1] == alt:
                    mva = pd.DataFrame(
                        [
                            [
                                s,
                                float(mi['paths']),
                                float(mi['mu']),
                                float(mi['adaptive']),
                                float(mi['m_bias']),
                                float(mi['a_bias']),
                                float(mi['m_bcpath']),
                                float(mi['a_bcpath'])
                            ]
                        ],
                        columns=col
                    )

                    m_v_a = merge(m_v_a, mva)

        m_v_a.to_csv(
            f'{mdir}'
            f'/mutation_vs_outcomes_for_{ref}>{alt}.csv',
            index=False
        )

    print(
        "frequencies of de novo & adaptive "
        "mutations of each type saved to folder"
    )


# Overall contribution of all available data toward rates,
# frequencies, and biases, for transition-transversion ratio
# and by individual mutation class:
def mcat_cont(
        species, data, all_mutations,
        bs, mvacol, tb, sdir
):

    for s in species:

        ma = pd.read_csv(
            f'{sdir}'
            f'/{s}_outcomes_model_vs_empirical_{data}.csv'
        )

        ti, tv = [0, 0, 0], [0, 0, 0]

        for n in range(len(ma)):

            mi = ma.iloc[n]
            ref, alt = mi['mutation'][0], mi['mutation'][-1]
            mdata = pd.DataFrame(
                [
                    [s, f'{ref}>{alt}'] + m_id_cont(
                        mi,
                        [0, 0, 0, 0, 0, 0, 0]
                    )
                ],
                columns=[mvacol[0], 'mutation'] + mvacol[1:]
            )

            all_mutations = merge(all_mutations, mdata)

            if alt in [bs[0], bs[2]]:
                ti = m_id_cont(mi, ti)
            else:
                tv = m_id_cont(mi, tv)

        tb = merge(
            tb,
            pd.DataFrame(
                [[s] + list(np.array(ti) / np.array(tv))],
                columns=mvacol[:2] + mvacol[4:6]
            )
        )

    # Return DataFrames of transition bias
    # and rates, frequencies & biases for all
    # six single-nucleotide mutation classes:
    return tb, all_mutations


# Calculate log likelihood by using a regression model
# to generate expected adaptive substitution biases &
# compare them to the actual (empirical or simulated)
# adaptive substitution biases:
def model_likelihood(mutations, adaptations, slp, itc):

    infer = np.array([])
    empir = np.array([])
    size = 0

    for m in range(len(mutations)):

        infer = np.append(
            infer, slp * np.array(mutations[m]) + np.array(itc[m]))
        empir = np.append(empir, np.array(adaptations[m]))
        size += len(adaptations[m])

    var = sum((infer - empir) ** 2) / size

    return (size / 2) * (-np.log(2 * np.pi) - np.log(var) - 1)


# Calculate regression details (slope, intercept, Pearson's r)
# for the (x,y) coordinates of a data set:
def least_sq(x, y):

    ave_x, ave_y = x.mean(), y.mean()
    cov_xy = np.cov(x, y)[0][1]
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)

    if var_y > 0:
        r = cov_xy / np.sqrt(var_x * var_y)
    else:
        r = "N/A"

    slope = cov_xy / var_x
    return [slope, ave_y - (slope * ave_x), r]


# Use a maximum-likelihood procedure to estimate
# the shared slope (beta) and intercepts (beta_0)
# for the six single-nucleotide mutation classes:
def max_likelihood(expected, observed, initial_intercepts):

    itl = np.concatenate(
        (
            np.array([0]),
            initial_intercepts
        )
    )

    def neg_l(x):
        return -model_likelihood(
            expected, observed,
            x[0], [x[1], x[2], x[3], x[4], x[5], x[6]],
        )

    m = minimize(
        neg_l,
        itl,
        method="Nelder-Mead",
        options={"adaptive": True}
    )

    return [m['x'][i] for i in range(7)]


# Use a maximum-likelihood procedure to estimate the slope (beta)
# and intercept (beta_0) for an input dataset of mutations and
# adaptive substitutions, taking into account heteroscedasticity,
# in other words species-specific errors in bootstrap variance,
# via a weighted total least squares regression:
def wlsq(
        m, a, m_n, a_n,
        itl, rescale, wts
):

    m, a = np.array(m), np.array(a)
    mcts = m_n * (m / (1 + m))
    acts = a_n * (a / (1 + a))
    m = np.log10(m).flatten()
    a = np.log10(a).flatten()
    avar = ((1 / acts) + (1 / (a_n - acts))) * rescale
    avar = avar.flatten()

    if wts == 'weighted_total':
        mvar = ((1 / mcts) + (1 / (m_n - mcts))) * rescale
        mvar = mvar.flatten()

    elif wts == 'weighted_adaptive':
        mvar = np.zeros(len(avar))

    else:
        raise RuntimeError(
            "Weights for weighted least-squares regression "
            "must be specified by a dataset, "
            "either 'adaptive' or 'total'"
        )

    def ssq(x):

        slp, itc = x[0], x[1]
        sqerr = (a - slp * m - itc) ** 2
        ptvar = ((slp ** 2) * mvar) + avar

        return sum((sqerr / ptvar) + np.log(ptvar))

    mxl = minimize(
        ssq, itl,
        method='Nelder-Mead',
        options={"adaptive": True}
    )

    return [mxl['x'][0], mxl['x'][1]]


# Calculate & save details of confidence estimates (top, bottom,
# and midpoint of 95% bootstrap confidence interval):
def confidence(
        rsdir, stdir,
        filep, stcol
):

    if not os.path.exists(stdir):
        os.makedirs(stdir)
    var, val, mdp, err, top, btm = [], [], [], [], [], []

    for (columnName, columnData) in pd.read_csv(
            f'{rsdir}'
            f'/{filep}',
            encoding='utf-8'
    ).items():

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

    pd.DataFrame(
        np.array([var, val, mdp, err, top, btm]).T,
        columns=stcol
    ).to_csv(f'{stdir}/{filep}')


# Perform and save statistics results across the simulated data sets:
def bootstats(
        rsdir, mdl,
        stcol, spec
):

    r, st = 'bootstrap_results', 'statistics'

    for root, dirs, files in os.walk(rsdir, topdown=True):

        sdr = root[:root.index(r)] + st + root[root.index(r) + len(r):]

        for file in files:

            if file != '.DS_Store':
                confidence(root, sdr, file, stcol)

    print(f"Bootstrap statistics saved for {spec} {mdl} results")


# Generate a simulated data set by
# resampling from a specified input data set:
def bootstrap(
        measure, dtyp, data,
        sp_mutn, sim_n,
        dsiz, paths
):

    boot = np.array(
        [
            np.array(
                [
                    np.ones(6) for s in range(len(sp_mutn))
                ]
            ) for r in range(sim_n)
        ]
    )

    for r in range(sim_n):

        for s in range(len(sp_mutn)):

            u, c = np.unique(
                np.random.choice(
                    6,
                    size=dsiz[s],
                    replace=True,
                    p=np.array(list(data[s] / sum(data[s])))
                ),
                return_counts=True
            )

            np.put(boot[r][s], u, c + 1)

            if dtyp == 'adaptive' and measure == 'rates':
                boot[r][s] /= paths[s]
            if dtyp == 'de_novo' and measure == 'freqs':
                boot[r][s] *= paths[s]

            boot[r][s] /= sum(boot[r][s])

    return boot


# Calculate various measures of bias from spectrum data:
def biasmeasures(data):

    g = sum(data[3:])

    return [
        (data[1] + data[3]) / (data[0] + data[2] + data[4] + data[5]),
        data / (1 - data),
        g / (1 - g),
        np.concatenate(
            (
                data[:3] / (1 - g - data[:3]),
                data[3:] / (g - data[3:])
            )
        )
    ]


# Save regression results for simulated and empirical data sets:
def bootresults(
        bdir, filename, empirical,
        simdata, sim_n, col
):

    tbl = pd.DataFrame(empirical + simdata, columns=col)

    tbl.insert(
        0,
        'data',
        pd.Series(
            ['empirical'] + [f'bootstrap_{r+1}' for r in range(sim_n)]
        )
    )

    tbl.to_csv(
        bdir + filename,
        encoding='utf-8',
        index=False
    )

    del tbl


# Perform regressions on a per-mutation-class basis for the six
# single-nucleotide classes, empirical and simulated data:
def pmbias(
        bdir, mutations, sp_mutn,
        sim_n, bias, loga, logm,
        boot, lboot, dsrc, btyp, bi
):

    e = [[list(bias[bi][i]) + least_sq(logm[i], loga[i])] for i in range(6)]
    if dsrc == "adaptive":
        n = [[list(boot[bi].T[r].T[i]) + least_sq(logm[i], lboot[i].T[r])
              for r in range(sim_n)] for i in range(6)]
    else:
        n = [[list(boot[bi].T[r].T[i]) + least_sq(lboot[i].T[r], loga[i])
              for r in range(sim_n)] for i in range(6)]

    for i in range(6):

        bootresults(
            bdir,
            f'/{mutations[i]}_{btyp}_bias_{dsrc}.csv',
            e[i], n[i], sim_n,
            list(sp_mutn['species']) + ['slope', 'intercept', 'r']
        )

    del e, n


# Perform regressions on consolidated data formats (transition bias &
# base-mutability bias), empirical and simulated data:
def cbias(
        bdir, sp_mutn, sim_n,
        bias, loga, logm, boot,
        lboot, dsrc, btyp, bi
):

    e = [list(bias[bi]) + least_sq(logm, loga)]
    if dsrc == "adaptive":
        n = [list(boot[bi].T[r]) + least_sq(logm, lboot[r])
             for r in range(sim_n)]
    else:
        n = [list(boot[bi].T[r]) + least_sq(lboot[r], loga)
             for r in range(sim_n)]

    bootresults(
        bdir,
        f'/{btyp}_bias_{dsrc}.csv',
        e, n, sim_n,
        list(sp_mutn['species']) + ['slope', 'intercept', 'r']
    )

    del e, n


# Perform single regression aggregating across intra- & inter-
# species variation in bias, empirical and simulated data:
def ixibias(
        bdir, sim_n, loga,
        logm, lboot, dsrc
):

    e = [least_sq(logm.flatten(), loga.flatten())]
    if dsrc == "adaptive":
        n = [least_sq(logm.flatten(), lboot.T[r].T.flatten())
             for r in range(sim_n)]
    else:
        n = [least_sq(lboot.T[r].T.flatten(), loga.flatten())
             for r in range(sim_n)]

    bootresults(
        bdir,
        f'/inter_x_intra_species_bias_{dsrc}.csv',
        e, n, sim_n,
        ['slope', 'intercept', 'r']
    )

    del e, n


# Perform regressions across mutation types,
# i.e. intra-species adaptive vs mutational bias:
def intraspbias(
        bdir, mutations, sp_mutn,
        sim_n, mub, adb, logm,
        loga, boot, lboot, dsrc
):

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

        bootresults(
            bdir,
            f'/{sp}_intra_sp_bias_{dsrc}.csv',
            e, n, sim_n,
            mutations + ['slope', 'intercept', 'r']
        )

        del e, n


# Perform regression using maximum-likelihood methodology
# to isolate interspecies variation in bias, with per-mutation
# proportionality constants, empirical and simulated data:
def ipmbias(
        bdir, mutations, sp_mutn,
        sim_n, bias, loga, logm, boot,
        lboot, dsrc, btyp, bi, altpath
):

    itl_icp = np.log10(np.ones(6) / altpath)
    ml = max_likelihood(logm, loga, itl_icp)
    n = [[], [], [], [], [], []]
    e = [[list(bias[bi][i - 1]) + [ml[0], ml[i]]] for i in range(1, 7)]
    if dsrc == "adaptive":

        for r in range(sim_n):

            ml_r = max_likelihood(logm, lboot[r].T, itl_icp)
            [n[i-1].append(
                list(boot[bi].T[r].T[i-1]) + [ml_r[0], ml_r[i]]
            )
                for i in range(1, 7)
            ]

    else:

        for r in range(sim_n):

            ml_r = max_likelihood(lboot[r].T, loga, itl_icp)
            [n[i-1].append(
                list(boot[bi].T[r].T[i-1]) + [ml_r[0], ml_r[i]]
            )
                for i in range(1, 7)
            ]

    for i in range(6):

        bootresults(
            bdir,
            f'/{mutations[i]}_intersp_panmut_{btyp}_bias_{dsrc}.csv',
            e[i], n[i], sim_n,
            list(sp_mutn['species']) + ['slope', 'intercept']
        )

    del e, n


# Find the slope (beta) and intercept (beta_0) for input dataset
# of mutations and adaptive substitutions, taking into account
# species-specific errors in bootstrap variance:
def errors_in_variables(
        rdir, bdir, sdir, jout, jdir,
        btyp, sim_n, sp_mutn, itl, wts
):

    if not os.path.exists(jdir):
        os.makedirs(jdir)

    m_n = np.array(sp_mutn['mu_counts'])
    a_n = np.array([
        sum(
            np.array(
                pd.read_csv(
                    f'{rdir}'
                    f'/by_species'
                    f'/{sp}_outcomes_model_vs_empirical_all_data.csv'
                )['adaptive_n']
            )
        ) for sp in list(sp_mutn['species'])
    ])

    adat, mdat = pd.DataFrame(), pd.DataFrame()
    rscl = 1 / (np.log(10) ** 2)

    for b_i in btyp:

        mdat = merge(
            mdat,
            pd.read_csv(
                f'{bdir}/de_novo/{sdir}/{b_i}_de_novo.csv',
                index_col='data'
            )
        )

        adat = merge(
            adat,
            pd.read_csv(
                f'{bdir}/adaptive/{sdir}/{b_i}_adaptive.csv',
                index_col='data'
            )
        )

    pt_wlsq = wlsq(
        mdat[mdat.columns[:-3]].loc['empirical'],
        adat[adat.columns[:-3]].loc['empirical'],
        m_n, a_n, itl, rscl, wts
    )

    pd.DataFrame(
        [pt_wlsq] + [
            wlsq(
                mdat[mdat.columns[:-3]].loc[f'bootstrap_{i}'],
                adat[adat.columns[:-3]].loc[f'bootstrap_{i}'],
                m_n, a_n, np.array(pt_wlsq), rscl, wts
            ) for i in range(1, sim_n + 1)
        ], columns=['slope', 'intercept']
    ).to_csv(f'{jdir}/{jout}_{wts}.csv', index=False)


# Analyze mutation vs adaptation data on bias,
# using empirical data and by simulating data:
def mva_analysis(
        rdir, measure, mutations, sp_mutn,
        sim_n, btypes, model, data
):

    ndir = f'{rdir}/analysis'
    scol = ['var', 'val', 'mid', 'err', '+95%', '-95%']

    pan_tax_paths = np.empty((0, 6), int)
    mdat = np.empty((0, 6), int)
    mdis = np.empty((0, 6), int)
    adat = np.empty((0, 6), int)
    rawdbsrc = np.empty((0, 6), int)
    ad_n = np.empty(0, int)

    for sp in list(sp_mutn['species']):

        spdat = pd.read_csv(
            f'{rdir}/by_species/{sp}_outcomes_model_vs_empirical_{data}.csv')

        rawad = np.array(spdat['adaptive_n'])
        ad_n = np.append(ad_n, np.array([round(rawad.sum())]))
        model_paths = np.array(spdat['paths'])
        pan_tax_paths = np.append(
            pan_tax_paths, np.array([model_paths]), axis=0)

        static_paths = np.array(
            pd.read_csv(
                f'./results'
                f'/results_with_stats_on_{sim_n}_sims'
                f'/bias_in_{measure}/{data}/by_species'
                f'/{sp}_outcomes_model_vs_empirical_{data}.csv'
            )['paths']
        )

        # Specify data source for generating simulated data
        # (empirical adaptive spectra or available mutational
        # paths for hypothesized model or null model, respectively):
        if model == 'model':

            sp_adat = np.array(spdat['adaptive'])
            rawdbsrc = np.append(
                rawdbsrc, np.array([rawad / rawad.sum()]), axis=0)

        elif model == 'null':

            sp_adat = static_paths
            rawdbsrc = np.append(
                rawdbsrc,
                np.array([static_paths / static_paths.sum()]), axis=0
            )

            if measure == 'rates':
                sp_adat /= model_paths
                sp_adat = sp_adat / sp_adat.sum()

        else:

            raise RuntimeError(
                'Model type must be specified '
                'as model or null'
            )

        # Compute mutation spectrum with pseudocounts
        # omitted, for bootstrapping:
        spmu = np.array([list(spdat['mu'])])
        if measure == 'freqs':
            spmu /= np.array([list(spdat['paths'])])
            spmu /= spmu.sum()

        m_n = int(
            sp_mutn.iloc[
                [list(sp_mutn['species']).index(sp)]]['mu_counts'].iloc[0]
        )

        spmu = (spmu * (m_n + 6)) - 1
        spmu = np.clip(spmu, 0, None)
        mdis = np.append(mdis, spmu / spmu.sum(), axis=0)
        mdat = np.append(mdat, np.array([list(spdat['mu'])]), axis=0)
        adat = np.append(adat, np.array([sp_adat]), axis=0)

    # Generate output folders and log-transformed data on mutation
    # and adaptive substitutions biases:
    for bt in btypes:

        folder(
            [
                f'{ndir}'
                f'/{model}'
                f'/bootstrap_results'
            ],
            [
                f'/adaptive/{bt}_bias',
                f'/de_novo/{bt}_bias',
                '/adaptive/within_species',
                '/de_novo/within_species'
            ]
        )

    bdir = f'{ndir}/{model}/bootstrap_results'
    mdb, adb = biasmeasures(mdat.T), biasmeasures(adat.T)
    lmti, lmgb = np.log10(mdb[0]), np.log10(mdb[1])
    lmbm, lmbc = np.log10(mdb[2]), np.log10(mdb[3])
    lati, lagb = np.log10(adb[0]), np.log10(adb[1])
    labm, labc = np.log10(adb[2]), np.log10(adb[3])

    # Simulate data sets and perform regression analyses:
    sim_source_data = [
        [
            "adaptive",
            rawdbsrc,
            ad_n,
            adb
        ], [
            "de_novo",
            mdis,
            list(sp_mutn['mu_counts']),
            mdb
        ]
    ]

    for src in sim_source_data:

        bt = biasmeasures(
            bootstrap(
                measure,
                src[0],
                src[1],
                sp_mutn,
                sim_n,
                src[2],
                pan_tax_paths
            ).T
        )

        btti, btgb = np.log10(bt[0].T), np.log10(bt[1])
        btbm, btbc = np.log10(bt[2].T), np.log10(bt[3])

        intraspbias(
            f'{bdir}/{src[0]}/within_species',
            mutations, sp_mutn, sim_n, mdb[1].T, adb[1].T,
            lmgb.T, lagb.T, bt[1].T, btgb.T, src[0]
        )

        ixibias(
            f'{bdir}/{src[0]}/general_bias',
            sim_n, lagb, lmgb, btgb, src[0]
        )

        cbias(
            f'{bdir}/{src[0]}/transition_bias',
            sp_mutn, sim_n, src[3], lati,
            lmti, bt, btti, src[0], "transition", 0
        )

        cbias(
            f'{bdir}/{src[0]}/base_mutability_bias',
            sp_mutn, sim_n, src[3], labm,
            lmbm, bt, btbm, src[0], "base_mutability", 2
        )

        pmbias(
            f'{bdir}/{src[0]}/general_bias',
            mutations, sp_mutn, sim_n, src[3], lagb,
            lmgb, bt, btgb, src[0], "general", 1
        )

        ipmbias(
            f'{bdir}/{src[0]}/general_bias',
            mutations, sp_mutn, sim_n, src[3], lagb,
            lmgb, bt, btgb.T, src[0], "general", 1, 5
        )

        pmbias(
            f'{bdir}/{src[0]}/alternate_base_bias',
            mutations, sp_mutn, sim_n, src[3], labc,
            lmbc, bt, btbc, src[0], "alternate_base", 3
        )

        ipmbias(
            f'{bdir}/{src[0]}/alternate_base_bias',
            mutations, sp_mutn, sim_n, src[3], labc,
            lmbc, bt, btbc.T, src[0], "alternate_base", 3, 2
        )

        del bt, btti, btgb, btbm, btbc

    # Generate and save statistical summary of regression results:
    bootstats(f'{bdir}/adaptive', model, scol, 'adaptive')
    bootstats(f'{bdir}/de_novo', model, scol, 'de_novo')

    # Errors-in-variables (weighted orthogonal distance regression)
    # analysis on all data, plus species permutations analysis,
    # and save results to output folder:
    if measure == 'rates' and data == 'all_data':

        for weights in ['weighted_adaptive', 'weighted_total']:

            errors_in_variables(
                rdir, bdir,
                'transition_bias',
                'transition_bias',
                f'{bdir}/{weights}/transition_bias',
                ['transition_bias'],
                sim_n, sp_mutn,
                np.array([0.0, np.log10(0.5)]),
                weights
            )

            errors_in_variables(
                rdir, bdir,
                'general_bias',
                'inter_x_intra_species_bias',
                f'{bdir}/{weights}/general_bias',
                [f'{m}_general_bias' for m in mutations],
                sim_n, sp_mutn,
                np.array([0.0, np.log10(0.2)]),
                weights
            )

            for b in mutations:

                errors_in_variables(
                    rdir, bdir,
                    'general_bias',
                    f'{b}_general_bias',
                    f'{bdir}/{weights}/general_bias',
                    [f'{b}_general_bias'],
                    sim_n, sp_mutn,
                    np.array([0.0, np.log10(0.2)]),
                    weights
                )

            bootstats(f'{bdir}/{weights}', model, scol, weights)

    print("Results from analysis of mutations vs "
          "adaptive outcomes saved to folder")


# Generate and save output tables containing original on
# frequencies & bias measurements for mutation and adaptive
# substitution data, plus detailed regression analyses for
# hypothesized vs null model:
def mva_compare(
        adir, rdir, measure, data, mutations, sp_mutn,
        sim_n, mvacol, m_spec, bp, bs, btypes,
        cntm, tangs_u, code, adata, dnmx,
        pseudorep, correct_for_pseudorep
):

    if not os.path.exists(adir):
        os.makedirs(adir)

    rdir = folder(
        [rdir],
        [
            '/by_mutation',
            '/by_species',
            '/analysis'
        ]
    )

    mdir = f'{rdir[0]}/by_mutation'
    sdir = f'{rdir[0]}/by_species'

    for s in sp_mutn['species']:

        adata[s] = adaptive_counter(
            s, mutations, bp, bs, [0, 0, 0, 0, 0, 0],
            correct_for_pseudorep, pseudorep
        )

        adaptations_per_species(
            sdir, measure, mutations, sp_mutn,
            cntm, s, data, bp, bs, m_spec,
            adata[s], tangs_u, code, dnmx
        )

    adp_per_mut_type(
        sdir, mdir, sp_mutn,
        data, mvacol,
        pd.DataFrame(
            mutations,
            columns=['mutation']
        )
    )

    for mdl in ['model', 'null']:

        mva_analysis(
            rdir[0], measure, mutations,
            sp_mutn, sim_n, btypes, mdl, data
        )

    c = mcat_cont(
        sp_mutn['species'], data,
        pd.DataFrame(columns=[mvacol[0], 'mutation'] + mvacol[1:]),
        bs, mvacol, pd.DataFrame(columns=mvacol), sdir
    )

    ti_bias = c[0]
    all_mutations = c[1]

    ti_bias.to_csv(
        f'{mdir}'
        f'/ti_bias_mutation_vs_outcomes.csv',
        index=False
    )

    all_mutations.to_csv(
        f'{mdir}'
        f'/all_mutation_classes_vs_outcomes.csv',
        index=False
    )

    adata.to_csv(
        f'{adir}'
        f'/adaptive_counts_{data}.csv',
        index=False
    )

    return print(f"RESULTS NOW COMPILED FOR {data}")