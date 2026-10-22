"""

Multi-taxon analysis of mutation-biased adaptation (mbamta)

This script reads the results of analyzing the effect of mutation bias on
adaptive substitutions, generated using mbamta_analyses.py, and generates
figures and tables summarizing & visualizing the results, in addition to
generating statistical significance tests using bootstraps and simulated
data generated using mbamta_analyses.py. See READ ME file for further
information and guidance.

"""


# Import packages that will be needed for this script:
import os
import warnings
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import gmean
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
import mbamta_analyses as mba


stcol = ['var', 'val', 'mid', 'err', '+95%', '-95%']

row_col_i = {
    'A>C': [1, 0], 'A>G': [0, 0], 'A>T': [2, 0],
    'G>A': [0, 1], 'G>C': [1, 1], 'G>T': [2, 1]
}

warnings.filterwarnings(
    "ignore", message="The figure layout has changed to tight")


# Append a row to a DataFrame:
def merge(append_to, appendix):

    if append_to.empty:
        append_to = appendix
    else:
        append_to = pd.concat([append_to, appendix])

    return append_to


# Reformat species names for graphing purposes:
def species_name(species):
    return species[0].upper() + f". {species[2:]}"


# Format numbers for visual presentation on graphs:
def num_form(nums):

    f = []

    for n in nums:

        if abs(n) < 0.01:
            f.append(
                np.format_float_scientific(n, precision=1, min_digits=1))
        else:
            f.append(round(n, 2))

    return f


# Calculate significance (p-value) by comparing
# the empirical point estimate to the results
# for simulated data under a null model:
def sig_simdat(
        empirical, simulated, tail, tails
):

    if tail == 'upper':
        overlap = sum(1 for s in simulated if s >= empirical)
    elif tail == 'lower':
        overlap = sum(1 for s in simulated if s <= empirical)
    else:
        raise RuntimeError(
            "Tail (upper or lower) for significance test not specified")

    if tails == 1:
        if overlap == 0:
            return f'<{1 / len(simulated)}'
        else:
            return f'{overlap / len(simulated)}'
    elif tails == 2:
        q = overlap / len(simulated)
        if min(q, 1 - q) == 0:
            return f'<{2 / len(simulated)}'
        else:
            return f'{2 * min(q, 1 - q)}'
    else:
        raise RuntimeError(
            "Tails must be set to 1 or 2 for significance test")


# Calculate bootstrap significance for data, append to list, return list:
def add_significance(filepath, sig, point, tail):

    boots = np.array(list(pd.read_csv(filepath)['slope']))[1:]
    sig.append(sig_simdat(point, boots, tail, 1))

    return sig


# Generate and save table of summary statistics on regression analyses:
def summary_tables(ardir, measure, data, mutations, bp):

    sdir = f'{ardir}/model/statistics/adaptive'
    nsdr = f'{ardir}/null/statistics/adaptive'

    biases = [
                 'transition',
                 'inter_x_intra_species'
             ] + [f'{m}_general' for m in mutations]

    subd = ['transition'] + ['general' for i in range(7)]

    lbls = [
               'Transition', 'Intra- & inter-species'
           ] + [f'{m[0]}:{bp[m[0]]}>{m[2]}:{bp[m[2]]}' for m in mutations]

    islp, nslp, iicp, ir, isig, btsig = [], [], [], [], [], []
    npsp = ['N/A', 'N/A']
    picp = ['N/A', 'N/A']
    pslp = ['N/A', 'N/A']
    psig = ['N/A', 'N/A']

    for bias in biases:

        isomfl = f'/{subd[biases.index(bias)]}_bias/{bias}_bias_adaptive.csv'

        # Append results and summary statistics to lists for the
        # single-slope regression that isolates interspecies variation:
        if biases.index(bias) >= 2:
            pan = pd.read_csv(
                f'{sdir}/general_bias'
                f'/{bias[0:3]}_intersp_panmut_general_bias_adaptive.csv'
            )

            pnn = pd.read_csv(
                f'{nsdr}/general_bias'
                f'/{bias[0:3]}_intersp_panmut_general_bias_adaptive.csv'
            )

            nsi = [i for i in range(len(pnn))
                   if 'slope' in list(pnn['var'])[i]][0]
            psi = [i for i in range(len(pan))
                   if 'slope' in list(pan['var'])[i]][0]
            pii = [i for i in range(len(pan))
                   if 'intercept' in list(pan['var'])[i]][0]

            pst = num_form(
                [list(pan['val'])[psi], list(pan['val'])[pii]] +
                [list(pan['-95%'])[psi], list(pan['+95%'])[psi]] +
                [list(pan['-95%'])[pii], list(pan['+95%'])[pii]]
            )

            nps = num_form([
                list(pnn['val'])[nsi],
                list(pnn['-95%'])[nsi],
                list(pnn['+95%'])[nsi]
            ])

            npsp.append(f'{nps[0]} ({nps[1]}\u2013{nps[2]})')
            pslp.append(f'{pst[0]} ({pst[2]}\u2013{pst[3]})')
            picp.append(f'{pst[1]} ({pst[4]}\u2013{pst[5]})')

            psig = add_significance(
                f'{ardir}/null/bootstrap_results/adaptive/general_bias'
                f'/{bias[0:3]}_intersp_panmut_general_bias_adaptive.csv',
                psig, list(pan['val'])[psi], 'upper'
            )

        isn = pd.read_csv(nsdr + isomfl)
        iso = pd.read_csv(sdir + isomfl)
        ni = list(iso['var']).index('slope')
        si = list(iso['var']).index('slope')
        ii = list(iso['var']).index('intercept')
        ri = list(iso['var']).index('r')

        ist = num_form(
            [
                list(iso['val'])[si],
                list(iso['val'])[ii],
                list(iso['val'])[ri]
            ] + [
                list(iso['-95%'])[si],
                list(iso['+95%'])[si]
            ] + [
                list(iso['-95%'])[ii],
                list(iso['+95%'])[ii]
            ] + [
                list(iso['-95%'])[ri],
                list(iso['+95%'])[ri]
            ]
        )

        nst = num_form([
            list(isn['val'])[ni],
            list(isn['-95%'])[ni],
            list(isn['+95%'])[ni]
        ])

        nslp.append(f'{nst[0]} ({nst[1]}\u2013{nst[2]})')
        islp.append(f'{ist[0]} ({ist[3]}\u2013{ist[4]})')
        iicp.append(f'{ist[1]} ({ist[5]}\u2013{ist[6]})')
        ir.append(f'{ist[2]} ({ist[7]}\u2013{ist[8]})')

        isig = add_significance(
            f'{ardir}/null/bootstrap_results/adaptive' + isomfl,
            isig, list(iso['val'])[si], 'upper'
        )

        # Append bootstrap significance of results to corresponding
        # lists for the rates-bias analysis that uses all data:
        if measure == 'rates' and data == 'all_data':

            if biases.index(bias) == 2:

                btsig = add_significance(
                    f'{ardir}/model/bootstrap_results/adaptive/general_bias'
                    f'/{bias[0:3]}_intersp_panmut_general_bias_adaptive.csv',
                    btsig, 0, 'lower'
                )

            btsig = add_significance(
                f'{ardir}/model/bootstrap_results/adaptive' + isomfl,
                btsig, 0, 'lower'
            )

    if measure == 'rates' and data == 'all_data':

        pd.DataFrame(
            zip(
                lbls[0:2] + ['Interspecies'] + lbls[2:],
                islp[0:2] + [pslp[3]] + islp[2:],
                btsig
            ), columns=['Type of bias', 'beta', 'Bootstrap significance']
        ).to_csv(f'{sdir}/bootstrap_significance.csv')

    pd.DataFrame(
        zip(
            lbls, islp, nslp, iicp, ir,
            isig, pslp, npsp, picp, psig
        ),
        columns=[
            'Type of bias',
            'beta',
            'null beta',
            'beta_0',
            'r',
            'significance',
            'panmut_beta',
            'panmut_null_beta',
            'panmut_beta_0',
            'panmut_significance'
        ]
    ).to_csv(f'{sdir}/regression_models_summary_stats.csv')


# Generate table of summary statistics for assessing effects of
# neutral contamination in adaptive substitutions data:
def contam_sumstats(iso, nul, effl, tbl, cols):

    si = list(iso['var']).index(effl)
    pfrm, ptxt = 0, sig_simdat(list(iso['val'])[si], nul, 'upper', 1)

    if '<' in ptxt:
        pfrm += 1

    tbl = merge(
        tbl,
        pd.DataFrame(
            zip(
                [list(iso['val'])[si]],
                [list(iso['mid'])[si]],
                [list(iso['err'])[si]],
                [float(ptxt[pfrm:])]
            ),
            columns=cols
        )
    )

    return tbl


# Calculate minimum, maximum, and range of axis values,
# for purposes of positioning labels on graph:
def axmeasures(axv):
    axmin, axmax = np.log10(axv[1])[0], np.log10(axv[1])[-1]
    return axmin, axmax, axmax - axmin


# Iterate through a list of species pairs, leaving out
# specified pairs of species pairs one at a time while
# performing regression analyses on the remaining data:
def iter_sans_species(rdir, nonind, ind, mutations):

    sspdir = f'{rdir}/{nonind}'

    if not os.path.exists(sspdir):
        os.makedirs(sspdir)

    files = [
                'all_mutation_classes_vs_outcomes.csv',
                'ti_bias_mutation_vs_outcomes.csv'
            ] + [
        f'mutation_vs_outcomes_for_{m}.csv'
        for m in mutations
    ]

    snfl = [
               'all_snm_bias_adaptive.csv',
               'transition_bias_adaptive.csv'
           ] + [
        f'{m}_bias_adaptive.csv'
        for m in mutations
    ]

    col = ['dataset', 'slope', 'intercept', 'r']
    sspres = [pd.DataFrame(columns=col) for i in range(8)]
    ipmbias = [pd.DataFrame(columns=col[:-1]) for i in range(6)]

    for sans in ind:

        mspr, aspr = [], []
        sanstxt = f'_{sans[0]}'
        if len(sans) > 1:
            for s in range(1, len(sans)):
                sanstxt += f'_{sans[s]}'

        for file in files:

            srcd = pd.read_csv(f'{rdir}/by_mutation/{file}')
            udat = pd.DataFrame()

            for i in srcd.index:

                if srcd.loc[i]['species'] not in sans:
                    udat = merge(udat, srcd.loc[i].to_frame().T)

            mb = np.log10(np.array(list(udat['m_bias'])))
            ab = np.log10(np.array(list(udat['a_bias'])))
            if files.index(file) >= 2:
                mspr.append(mb)
                aspr.append(ab)
            ls = mba.least_sq(mb, ab)

            lsrow = pd.DataFrame(
                zip(
                    [f'sans{sanstxt}'],
                    [ls[0]],
                    [ls[1]],
                    [ls[2]]
                ),
                columns=col
            )

            sspres[files.index(file)] = merge(
                sspres[files.index(file)],
                lsrow
            )

        ml = mba.max_likelihood(
            np.array(mspr),
            np.array(aspr),
            np.ones(6) / 5
        )

        for i in range(6):

            ipmbias[i] = merge(
                ipmbias[i],
                pd.DataFrame(
                    zip(
                        [f'sans{sanstxt}'],
                        [ml[0]],
                        [ml[i + 1]]
                    ),
                    columns=col[:-1]
                )
            )

        for file in files:

            sspres[files.index(file)].to_csv(
                f'{sspdir}'
                f'/{snfl[files.index(file)]}'
            )

        for i in range(6):

            ipmbias[i].to_csv(
                f'{sspdir}'
                f'/{mutations[i]}_panmut.csv'
            )


# Generate color-coded bar graphs of spectra
# for all species in the data set:
def fig_bar_graphs(
        rdir, spd, mutations, bp,
        measure, data, colors
):

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    titv = pd.read_csv(
        f'{rdir}/by_mutation/ti_bias_mutation_vs_outcomes.csv')
    spti = list(titv.sort_values(by='m_bias')['species'])
    sptx = [sp[0].upper() + '. ' + sp[2:] for sp in spti]

    axlbl = {
        'rates': [
            'Relative mutation rate',
            'Relative rate of adaptive change'
        ],
        'freqs': [
            'Frequency of ' + r'$de$ ' + r'$novo$ ' + 'missense mutations',
            'Fraction of adaptive changes'
        ]
    }

    for spec_l, spec_i, lbl in [['mu', 0, 'A'], ['adaptive', 1, 'B']]:

        spml = [
            list(pd.read_csv(
                f'{spd}/{sp}_outcomes_model_vs_empirical_{data}.csv'
            )[spec_l]) for sp in spti
        ]

        spm = np.array(spml).T
        btms, xtxt = np.zeros(len(spti)), -0.3

        for mi in ['A>G', 'G>A', 'A>C', 'A>T', 'G>C', 'G>T']:

            ytxt = 1.02
            if mi in ['A>G', 'A>C', 'G>C']:
                ytxt = 1.1
                if mi != 'A>G':
                    xtxt += 4.8

            ax[spec_i].text(
                xtxt, ytxt,
                f'{mi[0]}:{bp[mi[0]]}>{mi[-1]}:{bp[mi[-1]]}',
                fontsize=12, color=colors[mi],
                horizontalalignment='left'
            )

            ax[spec_i].bar(
                sptx, spm[mutations.index(mi)],
                bottom=btms, color=colors[mi], alpha=2 / 3
            )

            btms += spm[mutations.index(mi)]

        plt.setp(
            ax[spec_i].get_xticklabels(), rotation=45,
            ha='right', fontsize=10, fontstyle='italic'
        )

        plt.yticks(fontsize=10)
        ax[spec_i].margins(y=0)
        ax[spec_i].add_artist(
            lines.Line2D(
                [0, 4], [1.1, 1.1], color='black',
                linewidth=3
            )
        )

        ax[spec_i].set_ylabel(
            axlbl[measure][spec_i], fontsize=11, labelpad=4)

        for spine in ax[spec_i].spines.values():
            spine.set_linewidth(0.5)

        ax[spec_i].tick_params(
            which='both', width=0.5, length=3, right=True)

        ax[spec_i].text(
            -0.2, 1.2, lbl,
            transform=ax[spec_i].transAxes,
            fontsize=14,
            fontweight="bold",
            va="top"
        )

        ax[spec_i].set_box_aspect(1)

    plt.subplots_adjust(wspace=1/3)

    plt.savefig(
        f'{rdir}/figures/{measure}_spectra.pdf',
        bbox_inches='tight'
    )

    plt.close('all')


# Graph regressions on adaptive vs mutation bias across the six
# single-nucleotide mutation classes, on a per-species basis:
def fig_intra_species_regr(
        data, abias_lbl, andir, spdir,
        fdir, mutations, bp, sim_n,
        sp_mutn, colors, ucdsps, fd
):

    intracol = [
        "species",
        "beta",
        "null beta",
        "Pearson's r",
        "p"
    ]

    tbl = pd.DataFrame(columns=intracol)
    sn = len(list(sp_mutn['species']))
    row, col = 0, 0

    fig, ax = plt.subplots(
        fd[sn][0], fd[sn][1],
        figsize=(2 * fd[sn][1], 2 * fd[sn][0])
    )

    ax[0, fd[sn][2]].axis("off")

    for sp in list(sp_mutn['species']):

        biases = pd.read_csv(
            f'{spdir}'
            f'/{sp}_outcomes_model_vs_empirical_{data}.csv'
        )

        regr = pd.read_csv(
            f'{andir}'
            '/model'
            '/bootstrap_results'
            '/adaptive'
            '/within_species'
            f'/{sp}_intra_sp_bias_adaptive.csv'
        )

        nrgr = pd.read_csv(
            f'{andir}'
            '/null'
            '/bootstrap_results'
            '/adaptive'
            '/within_species'
            f'/{sp}_intra_sp_bias_adaptive.csv'
        )

        rsts = pd.read_csv(
            f'{andir}'
            '/model'
            '/statistics'
            '/adaptive'
            '/within_species'
            f'/{sp}_intra_sp_bias_adaptive.csv'
        )

        msts = pd.read_csv(
            f'{andir}'
            '/model'
            '/statistics'
            '/de_novo'
            '/within_species'
            f'/{sp}_intra_sp_bias_de_novo.csv'
        )

        nsts = pd.read_csv(
            f'{andir}'
            '/null'
            '/statistics'
            '/adaptive'
            '/within_species'
            f'/{sp}_intra_sp_bias_adaptive.csv'
        )

        si = list(rsts['var']).index('slope')
        ni = list(nsts['var']).index('slope')
        ri = list(rsts['var']).index('r')

        slp = num_form([
                list(rsts['val'])[si],
                list(rsts['-95%'])[si],
                list(rsts['+95%'])[si]
            ])

        nlp = num_form([
                list(nsts['val'])[ni],
                list(nsts['-95%'])[ni],
                list(nsts['+95%'])[ni]
            ])

        cor = num_form([
                list(rsts['val'])[ri],
                list(rsts['-95%'])[ri],
                list(rsts['+95%'])[ri]
            ])

        pval = sig_simdat(
            float(list(regr['slope'])[0]),
            list(nrgr['slope'])[1:],
            'upper', 1
        )

        if pval[0] == '<':
            ptxt = f"p{pval}"
        else:
            ptxt = f"p={pval}"

        tbl = merge(
            tbl,
            pd.DataFrame(
                zip(
                    [species_name(sp)],
                    [f'{slp[0]} ({slp[1]}\u2013{slp[2]})'],
                    [f'{nlp[0]} ({nlp[1]}\u2013{nlp[2]})'],
                    [f'{cor[0]} ({cor[1]}\u2013{cor[2]})'],
                    [ptxt]
                ),
                columns=intracol
            )
        )

        slps = np.array(regr['slope'])
        itc = np.array(regr['intercept'])
        mb = np.array(list(biases['m_bias']))
        ab = np.array(list(biases['a_bias']))

        axmax = np.log10(
            max(
                max(np.array(msts['+95%'])[:6]),
                max(np.array(rsts['+95%'])[:6])
            )
        )

        axmin = np.log10(
            min(
                min(np.array(msts['-95%'])[:6]),
                min(np.array(rsts['-95%'])[:6])
            )
        )

        margin = 0.1 * (axmax - axmin)
        axmin -= margin
        axmax += margin
        linrg = 10.0 ** np.array([axmin, axmax])

        ax[row, col].plot(
            linrg,
            10 ** ((np.log10(linrg) * slps[0]) + itc[0]),
            linewidth=1,
            color='black',
            zorder=10
        )

        for i in range(1, sim_n + 1):

            ax[row, col].plot(
                linrg,
                10 ** ((np.log10(linrg) * slps[i]) + itc[i]),
                linewidth=0.5,
                color='xkcd:silver',
                alpha=0.5,
                zorder=0
            )

        for mi in range(6):

            ax[row, col].text(
                mb[mi],
                ab[mi],
                ucdsps[mi],
                color=colors[mutations[mi]],
                fontsize=12,
                horizontalalignment='center',
                verticalalignment='center',
                zorder=20
            )

            ax[row, col].errorbar(
                np.array(msts['val'])[mi],
                np.array(rsts['mid'])[mi],
                yerr=np.array(rsts['err'])[mi],
                color=colors[mutations[mi]],
                linewidth=1,
                capsize=2,
                capthick=1,
                alpha=0.5,
                zorder=15
            )

            ax[row, col].errorbar(
                np.array(msts['mid'])[mi],
                np.array(rsts['val'])[mi],
                xerr=np.array(msts['err'])[mi],
                color=colors[mutations[mi]],
                linewidth=1,
                capsize=2,
                capthick=1,
                alpha=0.5,
                zorder=15
            )

        ax[row, col].text(
            0.02, 0.98,
            species_name(sp),
            fontsize=9,
            transform=ax[row, col].transAxes,
            fontstyle='italic',
            ha='left',
            va='top',
            zorder=10
        )

        ax[row, col].text(
            0.98, 0.02,
            ptxt,
            color='black',
            fontsize=8,
            transform=ax[row, col].transAxes,
            va='bottom',
            ha='right',
            zorder=15
        )

        ax[row, col].text(
            0.98, 0.1,
            f"$\u03B2$={slp[0]}",
            color='black',
            fontsize=8,
            transform=ax[row, col].transAxes,
            va='bottom',
            ha='right',
            zorder=20
        )

        ax[row, col].set_xlim(10 ** axmin, 10 ** axmax)
        ax[row, col].set_ylim(10 ** axmin, 10 ** axmax)
        ax[row, col].set_xscale('log')
        ax[row, col].set_yscale('log')
        ax[row, col].set_aspect('equal', adjustable='box')
        ax[row, col].xaxis.set_major_locator(
            LogLocator(base=10, subs=(1.0,)))
        ax[row, col].yaxis.set_major_locator(
            LogLocator(base=10, subs=(1.0,)))

        # Formatter: force plain decimal notation
        def tick_label_format(val, _):
            return f"{val:g}"

        ax[row, col].xaxis.set_major_formatter(
            FuncFormatter(tick_label_format))
        ax[row, col].yaxis.set_major_formatter(
            FuncFormatter(tick_label_format))
        ax[row, col].xaxis.set_minor_formatter(NullFormatter())
        ax[row, col].yaxis.set_minor_formatter(NullFormatter())

        for spine in ax[row, col].spines.values():
            spine.set_linewidth(0.5)

        ax[row, col].tick_params(
            which='both',
            width=0.5,
            top=False,
            right=False
        )

        ax[row, col].tick_params(
            which='minor',
            length=3,
            top=False,
            right=False
        )

        ax[row, col].tick_params(
            axis='both',
            which='major',
            length=6,
            top=False,
            right=False,
            labelsize=10
        )

        col += 1
        if [row, col] == [0, fd[sn][2]] or col == fd[sn][1]:
            row += 1
            col = 0

    # Create figure legend:
    y_pos = np.linspace(0.9, 0.1, 6)
    for i, y in enumerate(y_pos):

        ax[0, fd[sn][2]].text(
            0.15, y,
            f'{ucdsps[i]}',
            color=colors[mutations[i]],
            fontsize=12,
            ha='center',
            va='center'
        )

        ax[0, fd[sn][2]].text(
            0.25, y,
            f'{mutations[i][0]}:{bp[mutations[i][0]]}'
            f'>{mutations[i][-1]}:{bp[mutations[i][-1]]}',
            color=colors[mutations[i]],
            fontsize=12,
            ha='left',
            va='center'
        )

    fig.supxlabel('Mutation bias', fontsize=11)
    fig.supylabel(f'{abias_lbl}', fontsize=11)
    plt.tight_layout()
    plt.savefig(
        f'{fdir}'
        '/adaptive_vs_mutation_within_species.pdf',
        bbox_inches='tight'
    )

    plt.close('all')

    tbl.to_csv(
        f'{andir}'
        '/model'
        '/statistics'
        '/adaptive'
        '/within_species'
        '/intra_species_regressions_stats.csv'
    )


# Graph regression that aggregates
# intra- & interspecies variation in bias:
def fig_inter_x_intra_species(
        rdir, sfig, abias_lbl, mutations,
        mdl, btyp, spec, reanalysis
):

    if reanalysis:
        row = 1
    else:
        row = 0
    fig, ax = sfig
    sdir = f'{rdir}/analysis/{mdl}/statistics'
    bdir = f'{rdir}/analysis/{mdl}/bootstrap_results/{spec}/general_bias'
    regr = pd.read_csv(f'{bdir}/inter_x_intra_species_bias_{spec}.csv')

    regs = pd.read_csv(
        f'{sdir}/{spec}/general_bias'
        f'/inter_x_intra_species_bias_{spec}.csv'
    )

    msts = pd.DataFrame(columns=stcol)
    asts = pd.DataFrame(columns=stcol)
    ssl = regs.loc[regs['var'] == 'slope']
    sr = num_form([round(ssl[l].iloc[0], 2) for l in ['-95%', '+95%']])

    if abs(float(sr[0])) < 0.001:
        lo = '0.0'
    else:
        lo = sr[0]

    effect = round(ssl['val'].iloc[0], 2)
    txt = f"Effect ($\u03B2$): {effect} ({lo}\u2013{sr[1]})"

    for m in mutations:

        mst = pd.read_csv(
            f'{sdir}'
            '/de_novo'
            '/general_bias'
            f'/{m}_general_bias_de_novo.csv'
        )

        ast = pd.read_csv(
            f'{sdir}'
            '/adaptive'
            '/general_bias'
            f'/{m}_general_bias_adaptive.csv'
        )

        msts, asts = merge(msts, mst[:-3]), merge(asts, ast[:-3])

    mdpt, adpt = np.array(msts['val']), np.array(asts['val'])
    merr, aerr = np.array(msts['err']), np.array(asts['err'])
    mmid, amid = np.array(msts['mid']), np.array(asts['mid'])
    axmin = np.log10(np.min(pd.concat([msts['-95%'], asts['-95%']]).tolist()))
    axmax = np.log10(np.max(pd.concat([msts['+95%'], asts['+95%']]).tolist()))
    margin = 0.1 * (axmax - axmin)
    axmin -= margin
    axmax += margin

    ax[row, 0].scatter(
        mdpt,
        adpt,
        s=6,
        color='black',
        zorder=15
    )

    ax[row, 0].errorbar(
        mmid,
        adpt,
        xerr=merr,
        fmt='none',
        linewidth=0.5,
        color='black',
        alpha=0.5,
        zorder=15
    )

    ax[row, 0].errorbar(
        mdpt,
        amid,
        yerr=aerr,
        fmt='none',
        linewidth=0.5,
        color='black',
        alpha=0.5,
        zorder=15
    )

    ax[row, 0].set_xlabel(
        'Mutation bias',
        fontsize=11,
        labelpad=4
    )
    ax[row, 0].set_ylabel(
        abias_lbl,
        fontsize=11,
        labelpad=4
    )

    ax[row, 0].text(
        0.98, 0.02, txt,
        color='black',
        fontsize=10,
        transform=ax[row, 0].transAxes,
        va='bottom',
        ha='right',
        zorder=20
    )

    ax[row, 0].set_xlim(10 ** axmin, 10 ** axmax)
    ax[row, 0].set_ylim(10 ** axmin, 10 ** axmax)
    ax[row, 0].set_xscale('log')
    ax[row, 0].set_yscale('log')
    ax[row, 0].xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax[row, 0].yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))

    # Formatter: force plain decimal notation
    def tick_label_format(val, _):
        return f"{val:g}"

    ax[row, 0].xaxis.set_major_formatter(FuncFormatter(tick_label_format))
    ax[row, 0].yaxis.set_major_formatter(FuncFormatter(tick_label_format))

    for spine in ax[row, 0].spines.values():
        spine.set_linewidth(0.5)

    # Center the axes on x=1.0, y=1.0
    # if 1.0 is only major tick:
    axmin, axmax = ax[row, 0].get_xlim()
    majtix = ax[row, 0].get_xticks(minor=False)
    majtix = majtix[(majtix > axmin) & (majtix < axmax)]
    axmin, axmax = np.log10(axmin), np.log10(axmax)
    if len(majtix) == 3 and np.isclose(majtix[1], 0.1):
        excess = max(
            abs(axmax) + 1, abs(axmin) - 1
        ) - min(
            abs(axmax) + 1, abs(axmin) - 1
        )
        axmax -= excess / 2
        axmin -= excess / 2
        ax[row, 0].set_xlim(10 ** axmin, 10 ** axmax)
        ax[row, 0].set_ylim(10 ** axmin, 10 ** axmax)

    linrg = 10 ** np.array([axmin, axmax])
    slp, itc = np.array(regr['slope']), np.array(regr['intercept'])

    for bi in range(1, len(slp)):
        a_locr = 10 ** ((np.log10(linrg) * slp[bi]) + itc[bi])

        ax[row, 0].plot(
            linrg,
            a_locr,
            linewidth=0.5,
            color='xkcd:silver',
            alpha=0.5
        )

    ax[row, 0].plot(
        linrg,
        10 ** ((np.log10(linrg) * slp[0]) + itc[0]),
        linewidth=1,
        color='black',
        zorder=10
    )

    ax[row, 0].tick_params(
        which='both',
        width=0.5,
        top=False,
        right=False
    )

    ax[row, 0].tick_params(
        which='minor',
        length=3,
        top=False,
        right=False
    )

    ax[row, 0].tick_params(
        axis='both',
        which='major',
        length=6,
        top=False,
        right=False,
        labelsize=10
    )

    ax[row, 0].set_box_aspect(1)

    if reanalysis:
        ax[row, 0].text(
            -0.25, 1.05, 'B', transform=ax[row, 0].transAxes,
            fontsize=14, fontweight="bold", va="top"
        )
        return fig, ax
    else:
        plt.savefig(
            f'{rdir}'
            '/figures'
            f'/{mdl}'
            f'/inter_x_intra_species_bias_{spec}.pdf',
            bbox_inches='tight'
        )

        plt.close('all')


# Graph individual regressions on interspecies variation:
def fig_xspecies_regressions(
        rdir, sfig, abias_lbl, bp, mdl,
        spec, colors, ucdsym, reanalysis
):

    xf, ff, rf = 'analysis', 'figures', 'bootstrap_results'
    mf, af, sf = 'de_novo', 'adaptive', 'statistics'

    for pth, dirs, files in os.walk(rdir, topdown=True):

        if len(dirs) == 0 and not pth.endswith('/within_species'):

            s_i = pth.index(spec)
            mp = f'{pth[:s_i]}{mf}{pth[s_i + len(spec):]}'
            ap = f'{pth[:s_i]}{af}{pth[s_i + len(spec):]}'
            rp = f'{pth[:pth.index(sf)]}{rf}{pth[pth.index(sf) + len(sf):]}'
            fp = f'{pth[:pth.index(xf)]}{ff}/{mdl}{pth[s_i + len(spec):]}'

            if not os.path.exists(fp):
                os.makedirs(fp)

            if any('A>C' in file for file in files):
                fig, ax = plt.subplots(
                    3, 2, figsize=(10, 10),
                    gridspec_kw={'hspace': 0.075},
                    constrained_layout=True
                )
            else:
                if reanalysis:
                    fig, ax = sfig
                else:
                    fig, ax = plt.subplots(
                        figsize=(3 + 1/3, 10),
                        squeeze=False
                    )

            for file in files:

                if 'inter' not in file and 'intra' not in file:

                    if any('A>C' in f for f in files):
                        row = row_col_i[file[0:3]][0]
                        col = row_col_i[file[0:3]][1]
                    else:
                        row, col = 0, 0

                    btyp, sclr = file[:file.index('_bias_')], 'black'
                    rgst = pd.read_csv(f'{pth}/{file}')
                    slp = rgst.loc[rgst['var'] == 'slope']
                    sr = num_form([
                        round(slp[l].iloc[0], 2) for l in ['-95%', '+95%']])

                    if abs(float(sr[0])) < 0.001:
                        lo = '0.0'
                    else:
                        lo = sr[0]

                    effect = round(slp['val'].iloc[0], 2)
                    txt = f"Effect ($\u03B2$): {effect} ({lo}\u2013{sr[1]})"
                    mus = pd.read_csv(f'{mp}/{btyp}_bias_{mf}.csv')[:-3]
                    ads = pd.read_csv(f'{ap}/{btyp}_bias_{af}.csv')[:-3]
                    mdpt, adpt = np.array(mus['val']), np.array(ads['val'])
                    mmid, amid = np.array(mus['mid']), np.array(ads['mid'])
                    merr, aerr = np.array(mus['err']), np.array(ads['err'])
                    slp = np.array(pd.read_csv(f'{rp}/{file}')['slope'])
                    itc = np.array(pd.read_csv(f'{rp}/{file}')['intercept'])

                    if '>' in btyp:
                        refb, altb = btyp[0], btyp[2]
                        sclr = colors[f'{refb}>{altb}']

                        ax[row, col].text(
                            0.02, 0.98,
                            f'{refb}:{bp[refb]}>{altb}:{bp[altb]}',
                            color=colors[f'{refb}>{altb}'],
                            transform=ax[row, col].transAxes,
                            fontsize=12,
                            ha='left',
                            va='top'
                        )

                    for si in range(len(mus)):

                        ax[row, col].errorbar(
                            mmid[si],
                            adpt[si],
                            xerr=merr[si],
                            capsize=4,
                            capthick=1,
                            linewidth=1,
                            color=sclr,
                            alpha=0.25,
                            zorder=10
                        )

                        ax[row, col].errorbar(
                            mdpt[si],
                            amid[si],
                            yerr=aerr[si],
                            capsize=4,
                            capthick=1,
                            linewidth=1,
                            color=sclr,
                            alpha=0.25,
                            zorder=10
                        )

                        if len(mus) <= 20:

                            ax[row, col].text(
                                mdpt[si],
                                adpt[si],
                                ucdsym[si],
                                color=sclr,
                                fontsize=11,
                                horizontalalignment='center',
                                verticalalignment='center',
                                zorder=15
                            )

                            x_pos, y_start = 1.05, 1.0
                            y_step = 1.01 / len(mus)

                            ax[row, col].text(
                                x_pos,
                                y_start - si * y_step,
                                ucdsym[si],
                                color=sclr,
                                fontsize=10,
                                transform=ax[row, col].transAxes,
                                ha='center',
                                va='top'
                            )

                            ax[row, col].text(
                                x_pos + 0.04,
                                y_start - si * y_step,
                                species_name(mus['var'][si]),
                                fontsize=10,
                                fontstyle='italic',
                                transform=ax[row, col].transAxes,
                                ha='left',
                                va='top'
                            )

                        else:

                            ax[row, col].scatter(
                                mdpt[si],
                                adpt[si],
                                s=16,
                                color=sclr,
                                alpha=1 / 3,
                                zorder=15
                            )

                    ax[row, col].set_xlabel(
                        'Mutation bias',
                        fontsize=11,
                        labelpad=4
                    )

                    ax[row, col].set_ylabel(
                        abias_lbl,
                        fontsize=11,
                        labelpad=4
                    )

                    ax[row, col].text(
                        0.98, 0.02, txt,
                        color='black',
                        fontsize=10,
                        transform=ax[row, col].transAxes,
                        va='bottom',
                        ha='right',
                        zorder=20
                    )

                    axmin = np.log10(
                        np.min(
                            np.concatenate((
                                np.array(mus['-95%']),
                                np.array(ads['-95%'])
                            ))
                        )
                    )

                    axmax = np.log10(
                        np.max(
                            np.concatenate((
                                np.array(mus['+95%']),
                                np.array(ads['+95%'])
                            ))
                        )
                    )

                    margin = 0.15 * (axmax - axmin)
                    axmin -= margin
                    axmax += margin
                    ax[row, col].set_xlim(10 ** axmin, 10 ** axmax)
                    ax[row, col].set_ylim(10 ** axmin, 10 ** axmax)
                    ax[row, col].set_xscale('log')
                    ax[row, col].set_yscale('log')

                    ax[row, col].xaxis.set_major_locator(
                        LogLocator(base=10, subs=(1.0,)))
                    ax[row, col].yaxis.set_major_locator(
                        LogLocator(base=10, subs=(1.0,)))
                    ax[row, col].xaxis.set_minor_formatter(NullFormatter())
                    ax[row, col].yaxis.set_minor_formatter(NullFormatter())

                    # Add min and max minor tick labels
                    # if only 1 labeled major tick falls
                    # within the axis range:
                    def label_min_tix(
                            ax, center=1.0,
                            margin_factor=0.9
                    ):

                        # Get axis min and max values:
                        axmin, axmax = ax.get_xlim()

                        # Get major tick locations:
                        majtix = ax.get_xticks(minor=False)
                        majtix = majtix[(majtix > axmin) & (majtix < axmax)]

                        # Only proceed if there is
                        # only a tick label at x=1:
                        if len(majtix) != 1:
                            return
                        if not np.isclose(majtix[0], center):
                            return

                        # Get minor tick locations:
                        tix = ax.get_xticks(minor=True)
                        tix = tix[(tix > axmin) & (tix < axmax)]
                        d = np.log10(tix)
                        left = d[d < 0]
                        right = d[d > 0]
                        if len(left) == 0:
                            return
                        if len(right) == 0:
                            return

                        # Get min and max minor ticks that are
                        # symmetrical (equidistant on log scale)
                        # around x=1:
                        maxsym = min(-left.min(), right.max())

                        symtix = [
                            10 ** left[np.argmin(np.abs(left + maxsym))],
                            10 ** right[np.argmin(np.abs(right - maxsym))]
                        ]

                        # Add min and max minor tick labels:
                        for t in symtix:

                            ax.text(
                                t, axmin * margin_factor,
                                f"{t:g}", ha='center', va='top',
                                fontsize=10
                            )

                            ax.text(
                                axmin * margin_factor, t,
                                f"{t:g}", ha='right', va='center',
                                fontsize=10
                            )

                        ax.yaxis.set_label_coords(-0.12, 0.5)

                    # Formatter: force plain decimal notation
                    def tick_label_format(val, _):
                        return f"{val:g}"

                    ax[row, col].xaxis.set_major_formatter(
                        FuncFormatter(tick_label_format))
                    ax[row, col].yaxis.set_major_formatter(
                        FuncFormatter(tick_label_format))

                    for spine in ax[row, col].spines.values():
                        spine.set_linewidth(0.5)

                    # Center the axes on x=1.0, y=1.0
                    # if 1.0 is only major tick:
                    axmin, axmax = ax[row, col].get_xlim()
                    majtix = ax[row, col].get_xticks(minor=False)
                    majtix = majtix[(majtix > axmin) & (majtix < axmax)]
                    axmin, axmax = np.log10(axmin), np.log10(axmax)
                    if len(majtix) == 1 and np.isclose(majtix[0], 1.0):
                        excess = max(
                            abs(axmax), abs(axmin)
                        ) - min(
                            abs(axmax), abs(axmin)
                        )
                        axmax -= excess / 2
                        axmin -= excess / 2
                        ax[row, col].set_xlim(10 ** axmin, 10 ** axmax)
                        ax[row, col].set_ylim(10 ** axmin, 10 ** axmax)

                    label_min_tix(ax[row, col])

                    linrg = 10 ** np.array([axmin, axmax])
                    for bi in range(1, len(slp)):
                        a_locr = 10 ** (
                                (np.log10(linrg) * slp[bi]) + itc[bi])

                        ax[row, col].plot(
                            linrg,
                            a_locr,
                            linewidth=0.5,
                            color='xkcd:silver',
                            alpha=0.5
                        )

                    ax[row, col].plot(
                        linrg,
                        10 ** ((np.log10(linrg) * slp[0]) + itc[0]),
                        linewidth=1,
                        color='black',
                        zorder=5
                    )

                    ax[row, col].tick_params(
                        which='both',
                        width=0.5,
                        top=True,
                        right=True
                    )

                    ax[row, col].tick_params(
                        which='minor',
                        length=3,
                        top=False,
                        right=False
                    )

                    ax[row, col].tick_params(
                        axis='both',
                        which='major',
                        length=6,
                        top=False,
                        right=False,
                        labelsize=10
                    )

                    ax[row, col].set_box_aspect(1)

            if reanalysis:
                ax[0, 0].text(
                    -0.25, 1.05, 'A', transform=ax[0, 0].transAxes,
                    fontsize=14, fontweight="bold", va="top"
                )
                return fig, ax
            elif any('A>C' in file for file in files):
                fig.savefig(f'{fp}/{pth[s_i + len(spec)+1:]}_{spec}.pdf')
                plt.close(fig)
            else:
                fig.savefig(
                    f'{fp}/{pth[s_i + len(spec) + 1:]}_{spec}.pdf',
                    bbox_inches='tight'
                )
                plt.close(fig)


# Plot adaptive substitution bias against mutation bias
# for each of the six mutation classes on a single plot:
def fig_combined_regressions(
        rdir, sfig, abias_lbl, sdir, mutations,
        mdl, spec, colors, markers, reanalysis
):

    fig, ax = sfig
    if reanalysis:
        row = 2
    else:
        row = 0

    mf, af, sf = 'de_novo', 'adaptive', 'statistics'
    msts = pd.DataFrame(columns=stcol)
    asts = pd.DataFrame(columns=stcol)
    mrd, ard = {}, {}
    maxm = []

    for m in mutations:

        mst = pd.read_csv(
            f'{sdir}'
            '/de_novo'
            '/general_bias'
            f'/{m}_general_bias_de_novo.csv'
        )[:-3]

        ast = pd.read_csv(
            f'{sdir}'
            '/adaptive'
            '/general_bias'
            f'/{m}_general_bias_adaptive.csv'
        )[:-3]

        msts, asts = merge(msts, mst), merge(asts, ast)
        mrd[m] = [float(min(mst['-95%'])), float(max(mst['+95%']))]
        ard[m] = [float(min(ast['-95%'])), float(max(ast['+95%']))]

    axmin = np.log10(
        np.min(pd.concat([msts['-95%'], asts['-95%']]).tolist()))
    axmax = np.log10(
        np.max(pd.concat([msts['+95%'], asts['+95%']]).tolist()))
    margin = 0.05 * (axmax - axmin)
    axmin -= margin
    axmax += margin

    for pth, dirs, files in os.walk(
            f'{sdir}/{spec}/general_bias',
            topdown=True
    ):

        for file in files:

            rgst = pd.read_csv(f'{pth}/{file}')
            slp_i = rgst.loc[rgst['var'] == 'slope']
            itc_i = rgst.loc[rgst['var'] == 'intercept']
            sl = float(slp_i['val'].iloc[0])
            ic = float(itc_i['val'].iloc[0])

            if 'inter' not in file:
                m = file[:3]

                mus = pd.read_csv(
                    f'{sdir}'
                    f'/{mf}'
                    '/general_bias'
                    f'/{file[:17]}{mf}.csv'
                )[:-3]

                ads = pd.read_csv(
                    f'{sdir}'
                    f'/{af}'
                    '/general_bias'
                    f'/{file[:17]}{af}.csv'
                )[:-3]

                mdpt, adpt = np.array(mus['val']), np.array(ads['val'])
                mmid, amid = np.array(mus['mid']), np.array(ads['mid'])
                merr, aerr = np.array(mus['err']), np.array(ads['err'])

                if markers:
                    data_alpha = 0.2
                    data_edge_alpha = 0.5
                else:
                    data_alpha = 0.25
                    data_edge_alpha = 1

                ax[row, 0].scatter(
                    mdpt,
                    adpt,
                    s=6,
                    facecolor=colors[m],
                    alpha=data_alpha,
                    zorder=10
                )

                ax[row, 0].scatter(
                    mdpt,
                    adpt,
                    s=6,
                    facecolor='none',
                    linewidth=1/3,
                    edgecolor=colors[m],
                    alpha=data_edge_alpha,
                    zorder=15
                )

                if markers:
                    ax[row, 0].scatter(
                        gmean(mdpt),
                        gmean(adpt),
                        s=24,
                        marker='X',
                        linewidth=0.1,
                        color='k',
                        zorder=30
                    )

                for si in range(len(mus)):
                    ax[row, 0].errorbar(
                        mmid[si],
                        adpt[si],
                        xerr=merr[si],
                        linewidth=2/3,
                        color=colors[m],
                        alpha=data_alpha,
                        zorder=5
                    )

                    ax[row, 0].errorbar(
                        mdpt[si],
                        amid[si],
                        yerr=aerr[si],
                        linewidth=2/3,
                        color=colors[m],
                        alpha=data_alpha,
                        zorder=5
                    )

                mblim = np.array([mrd[m][0], mrd[m][1]])
                ablim = np.array([ard[m][0], ard[m][1]])
                if mdl == 'null' or abs(sl) <= 10e-9:
                    xrg, yrg = mblim, 10 ** ((np.log10(mblim) * sl) + ic)
                else:
                    arg = (np.log10(mblim) * sl) + ic
                    mrg = (np.log10(ablim) - ic) / sl
                    if max(mrg) - min(mrg) >= max(arg) - min(arg):
                        xrg, yrg = mblim, 10 ** arg
                    else:
                        xrg, yrg = 10 ** mrg, ablim

                maxm.append(max(xrg))
                rmin = min([min(np.log10(xrg)), min(np.log10(yrg))])
                rmax = max([max(np.log10(xrg)), max(np.log10(yrg))])

                if rmin < axmin:
                    axmin = rmin - (0.05 * (rmax - rmin))
                if rmax > axmax:
                    axmax = rmax + (0.1 * (rmax - rmin))

                if m == 'G>A' or m == 'A>G':
                    zlevel = 25
                else:
                    zlevel = 20

                ax[row, 0].plot(
                    xrg,
                    yrg,
                    linewidth=2,
                    color=colors[m],
                    alpha=2/3,
                    zorder=zlevel
                )

    m_labels = [
        'G:C>A:T', 'A:T>G:C', 'G:C>T:A',
        'G:C>C:G', 'A:T>T:A', 'A:T>C:G'
    ]

    x_pos, y_start, y_step = 1.04, 0.98, 0.164
    bstart, pstart = 0.93, 0.885

    for i, text in enumerate(m_labels):

        m = f"{m_labels[i][0]}>{m_labels[i][4]}"

        ax[row, 0].text(
            x_pos,
            y_start - i * y_step,
            text,
            color=colors[m],
            transform=ax[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight='bold'
        )

        sts = pd.read_csv(
            f'{rdir}/analysis/model/statistics'
            '/adaptive/regression_models_summary_stats.csv'
        )

        nslps = list(
            pd.read_csv(
                f'{rdir}'
                '/analysis'
                '/null'
                '/bootstrap_results'
                f'/{spec}'
                '/general_bias'
                f'/{m}_general_bias_{spec}.csv'
            )['slope']
        )[1:]

        if spec == 'weighted_adaptive' or spec == 'weighted_total':

            mstats = pd.read_csv(
                f'{sdir}'
                f'/{spec}'
                '/general_bias'
                f'/{m}_general_bias_{spec}.csv'
            )

            slp_i = mstats.loc[mstats['var'] == 'slope']
            slp_p = float(slp_i['val'].iloc[0])
            slp_l = round(float(slp_i['-95%'].iloc[0]), 2)
            slp_h = round(float(slp_i['+95%'].iloc[0]), 2)
            sig = sig_simdat(slp_p, nslps, 'upper', 1)
            slp_p = round(slp_p, 2)
            eff = f"{slp_p} ({slp_l}\u2013{slp_h})"

        else:

            eff = str(
                sts.loc[
                    sts['Type of bias'] == m_labels[i],
                    'beta'
                ].iloc[0]
            )

            sig = str(
                sts.loc[
                    sts['Type of bias'] == m_labels[i],
                    'significance'
                ].iloc[0]
            )

        if sig[0] == '<':
            sig = f'p{sig}'
        else:
            sig = f'p={sig}'

        ax[row, 0].text(
            x_pos,
            bstart - i * y_step,
            f"$\u03B2$ = {eff}",
            transform=ax[row, 0].transAxes,
            color=colors[m],
            ha="left",
            va="top",
            fontsize=8
        )

        ax[row, 0].text(
            x_pos,
            pstart - i * y_step,
            f"{sig}",
            transform=ax[row, 0].transAxes,
            color=colors[m],
            ha="left",
            va="top",
            fontsize=8
        )

    ax[row, 0].set_xlabel(
        'Mutation bias',
        fontsize=11,
        labelpad=4
    )
    ax[row, 0].set_ylabel(
        abias_lbl,
        fontsize=11,
        labelpad=4
    )

    ax[row, 0].set_xlim(10**axmin, 10**axmax)
    ax[row, 0].set_ylim(10**axmin, 10**axmax)
    ax[row, 0].set_xscale('log')
    ax[row, 0].set_yscale('log')
    ax[row, 0].xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax[row, 0].yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))

    # Formatter: force plain decimal notation
    def tick_label_format(val, _):
        return f"{val:g}"

    ax[row, 0].xaxis.set_major_formatter(FuncFormatter(tick_label_format))
    ax[row, 0].yaxis.set_major_formatter(FuncFormatter(tick_label_format))

    for spine in ax[row, 0].spines.values():
        spine.set_linewidth(0.5)

    # Center the axes on x=1.0, y=1.0
    # if 1.0 is only major tick:
    axmin, axmax = ax[row, 0].get_xlim()
    majtix = ax[row, 0].get_xticks(minor=False)
    majtix = majtix[(majtix > axmin) & (majtix < axmax)]
    axmin, axmax = np.log10(axmin), np.log10(axmax)
    if len(majtix) == 3 and np.isclose(majtix[1], 0.1):
        excess = max(
            abs(axmax) + 1, abs(axmin) - 1
        ) - min(
            abs(axmax) + 1, abs(axmin) - 1
        )
        axmax -= excess / 2
        axmin -= excess / 2
        ax[row, 0].set_xlim(10 ** axmin, 10 ** axmax)
        ax[row, 0].set_ylim(10 ** axmin, 10 ** axmax)

    if markers:
        ax[row, 0].plot(
            np.array([10 ** axmin, 10 ** axmax]),
            np.array([10 ** axmin, 10 ** axmax]),
            linestyle=':',
            linewidth=1,
            color='k',
            zorder=0
        )

    ax[row, 0].tick_params(
        which='both',
        width=0.5,
        top=False,
        right=False
    )

    ax[row, 0].tick_params(
        which='minor',
        length=3,
        top=False,
        right=False
    )

    ax[row, 0].tick_params(
        axis='both',
        which='major',
        length=6,
        top=False,
        right=False,
        labelsize=10
    )

    ax[row, 0].set_box_aspect(1)

    if not os.path.exists(f'{rdir}/figures/{mdl}'):
        os.makedirs(f'{rdir}/figures/{mdl}')

    if reanalysis:
        ax[row, 0].text(
            -0.25, 1.05, 'C', transform=ax[row, 0].transAxes,
            fontsize=14, fontweight="bold", va="top"
        )
        return fig, ax
    else:
        if markers:
            filenm = f'combined_regressions_{spec}_with_beta_1_marked.pdf'
        else:
            filenm = f'combined_regressions_by_mutation_{spec}.pdf'
        plt.savefig(f'{rdir}/figures/{mdl}/{filenm}', bbox_inches='tight')
        plt.close('all')


# Graph regressions that isolate variation across species
# in mutation bias that aggregates the effect of bias
# from all six mutation classes:
def fig_multi_regressions(
        rdir, abias_lbl, mutations, bp,
        sp_n, mdl, btyp, spec,
        colors, ucdsym
):

    fdir = rdir[:rdir.index('analysis')] + f'figures/{mdl}/{btyp}_bias'
    fig, ax = plt.subplots(
        3, 2, figsize=(10, 10),
        gridspec_kw={'hspace': 0.075},
        constrained_layout=True
    )

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    sdir = f'{rdir}/{mdl}/statistics'
    bdir = f'{rdir}/{mdl}/bootstrap_results/{spec}'
    msts = pd.DataFrame(columns=stcol)
    asts = pd.DataFrame(columns=stcol)

    for m in mutations:
        mst = pd.read_csv(
            f'{sdir}'
            '/de_novo'
            f'/{btyp}_bias'
            f'/{m}_intersp_panmut_{btyp}_bias_de_novo.csv'
        )

        ast = pd.read_csv(
            f'{sdir}'
            '/adaptive'
            f'/{btyp}_bias'
            f'/{m}_intersp_panmut_{btyp}_bias_adaptive.csv'
        )

        msts, asts = merge(msts, mst[:-2]), merge(asts, ast[:-2])

    mdpt, adpt = np.array(msts['val']), np.array(asts['val'])
    merr, aerr = np.array(msts['err']), np.array(asts['err'])
    mmid, amid = np.array(msts['mid']), np.array(asts['mid'])

    axmin = np.log10(
        np.min(pd.concat([msts['-95%'], asts['-95%']]).tolist()))
    axmax = np.log10(
        np.max(pd.concat([msts['+95%'], asts['+95%']]).tolist()))

    margin = 0.1 * (axmax - axmin)
    axmin -= margin
    axmax += margin
    linrg = 10 ** np.array([axmin, axmax])

    for m in mutations:

        row, col = row_col_i[m][0], row_col_i[m][1]

        mdx = int(round(mutations.index(m) * sp_n))
        mlbl = f'{m[0]}:{bp[m[0]]}>{m[-1]}:{bp[m[-1]]}'
        mdi = pd.read_csv(
            f'{sdir}'
            '/de_novo'
            f'/{btyp}_bias'
            f'/{m}_intersp_panmut_{btyp}_bias_de_novo.csv'
        )

        adi = pd.read_csv(
            f'{sdir}'
            '/adaptive'
            f'/{btyp}_bias'
            f'/{m}_intersp_panmut_{btyp}_bias_adaptive.csv'
        )

        alog = pd.read_csv(
            f'{sdir}'
            '/adaptive'
            f'/{btyp}_bias'
            f'/{m}_intersp_panmut_{btyp}_bias_adaptive.csv'
        )

        ssl = alog.loc[alog['var'] == 'slope']
        sr = num_form([round(ssl[l].iloc[0], 2) for l in ['-95%', '+95%']])

        if abs(float(sr[0])) < 0.001:
            lo = '0.0'
        else:
            lo = sr[0]

        effect = round(ssl['val'].iloc[0], 2)
        txt = f"Effect ($\u03B2$): {effect} ({lo}\u2013{sr[1]})"
        mdv, adv = np.array(mdi['val'][:-2]), np.array(adi['val'][:-2])

        rgs = pd.read_csv(
            f'{bdir}'
            f'/{btyp}_bias'
            f'/{m}_intersp_panmut_{btyp}_bias_{spec}.csv'
        )

        slp = np.array(rgs['slope'])
        itc = np.array(rgs['intercept'])

        for bi in range(1, len(slp)):

            a_locr = 10 ** ((np.log10(linrg) * slp[bi]) + itc[bi])
            ax[row, col].plot(
                linrg,
                a_locr,
                linewidth=0.5,
                color='xkcd:silver',
                alpha=0.5
            )

        ax[row, col].errorbar(
            mmid,
            adpt,
            xerr=merr,
            fmt='none',
            linewidth=1,
            color='xkcd:light grey',
            zorder=5
        )

        ax[row, col].errorbar(
            mdpt,
            amid,
            yerr=aerr,
            fmt='none',
            linewidth=1,
            color='xkcd:light grey',
            zorder=5
        )

        ax[row, col].scatter(
            mdpt,
            adpt,
            s=16,
            color='xkcd:light grey',
            zorder=5
        )

        ax[row, col].plot(
            linrg,
            10 ** ((np.log10(linrg) * slp[0]) + itc[0]),
            linewidth=1,
            color='black',
            zorder=10
        )

        ax[row, col].errorbar(
            mmid[mdx:mdx + sp_n],
            adpt[mdx:mdx + sp_n],
            xerr=merr[mdx:mdx + sp_n],
            fmt='none',
            linewidth=1,
            capsize=4,
            capthick=1,
            color=colors[m],
            alpha=0.5,
            zorder=15
        )

        ax[row, col].errorbar(
            mdpt[mdx:mdx + sp_n],
            amid[mdx:mdx + sp_n],
            yerr=aerr[mdx:mdx + sp_n],
            fmt='none',
            linewidth=1,
            capsize=4,
            capthick=1,
            color=colors[m],
            alpha=0.5,
            zorder=15
        )

        for si in range(len(mdv)):

            if len(mdv) <= 20:

                ax[row, col].text(
                    mdv[si],
                    adv[si],
                    ucdsym[si],
                    color=colors[m],
                    fontsize=11,
                    horizontalalignment='center',
                    verticalalignment='center',
                    zorder=15
                )

                x_pos, y_start = 1.05, 1.0
                y_step = 1.01 / len(mdv)

                ax[row, col].text(
                    x_pos,
                    y_start - si * y_step,
                    ucdsym[si],
                    color=colors[m],
                    fontsize=10,
                    transform=ax[row, col].transAxes,
                    ha='center',
                    va='top'
                )

                ax[row, col].text(
                    x_pos + 0.04,
                    y_start - si * y_step,
                    species_name(mdi['var'][si]),
                    fontsize=10,
                    fontstyle='italic',
                    transform=ax[row, col].transAxes,
                    ha='left',
                    va='top'
                )

            else:
                ax[row, col].scatter(
                    mdv,
                    adv,
                    s=16,
                    color=colors[m],
                    edgecolors='black',
                    zorder=20
                )

        if m == 'A>G':

            ax[row, col].text(
                0.98, 0.02, txt,
                color='black',
                fontsize=10,
                transform=ax[row, col].transAxes,
                va='bottom',
                ha='right',
                zorder=20
            )

        ax[row, col].set_xlabel(
            'Mutation bias',
            fontsize=11,
            labelpad=4
        )

        ax[row, col].set_ylabel(
            abias_lbl,
            fontsize=11,
            labelpad=4
        )

        ax[row, col].text(
            0.02, 0.98, mlbl,
            transform=ax[row, col].transAxes,
            color=colors[m],
            fontsize=12,
            ha='left',
            va='top'
        )

        ax[row, col].set_xlim(10 ** axmin, 10 ** axmax)
        ax[row, col].set_ylim(10 ** axmin, 10 ** axmax)
        ax[row, col].set_xscale('log')
        ax[row, col].set_yscale('log')
        ax[row, col].xaxis.set_major_locator(
            LogLocator(base=10, subs=(1.0,)))
        ax[row, col].yaxis.set_major_locator(
            LogLocator(base=10, subs=(1.0,)))

        # Formatter: force plain decimal notation
        def tick_label_format(val, _):
            return f"{val:g}"

        ax[row, col].xaxis.set_major_formatter(
            FuncFormatter(tick_label_format))
        ax[row, col].yaxis.set_major_formatter(
            FuncFormatter(tick_label_format))

        for spine in ax[row, col].spines.values():
            spine.set_linewidth(0.5)

        ax[row, col].tick_params(
            which='both',
            width=0.5,
            top=False,
            right=False
        )

        ax[row, col].tick_params(
            which='minor',
            length=3,
            top=False,
            right=False
        )

        ax[row, col].tick_params(
            axis='both',
            which='major',
            length=6,
            top=False,
            right=False,
            labelsize=10
        )

        ax[row, col].set_box_aspect(1)

    plt.savefig(
        f'{fdir}/{mdl}_interspecific_{btyp}_bias_{spec}.pdf',
        bbox_inches='tight'
    )

    plt.close('all')


# Graph the regressions resulting from
# leaving each species out one at a time:
def fig_sans_species(
        rdir, abias_lbl, sspdir, fdir,
        mutations, bp, colors
):

    if not os.path.exists(f'{fdir}/sans_species'):
        os.makedirs(f'{fdir}/sans_species')

    for root, dirs, files in os.walk(sspdir, topdown=True):

        for file in files:

            if file == 'pred_vs_obs_permutation_tests.csv':
                continue

            regrs = pd.read_csv(f'{sspdir}/{file}')
            scol = [col for col in regrs.columns if 'slope' in col]
            icol = [col for col in regrs.columns if 'intercept' in col]
            slopes, intercepts = regrs[scol[0]], regrs[icol[0]]
            sdir = f'{rdir}/analysis/model/statistics'
            sts = pd.DataFrame(columns=stcol)

            if file[:3] in mutations:
                m = file[:3]
                rclr = colors[file[:3]]
                if 'panmut' in file:
                    res = pd.read_csv(
                        f'{sdir}'
                        '/adaptive/general_bias'
                        f'/{m}_intersp_panmut_general_bias_adaptive.csv'
                    )

                else:
                    res = pd.read_csv(
                        f'{sdir}'
                        '/adaptive/general_bias'
                        f'/{m}_general_bias_adaptive.csv'
                    )

            else:
                rclr = 'grey'
                if 'transition' in file:
                    res = pd.read_csv(
                        f'{sdir}'
                        '/adaptive'
                        f'/transition_bias/{file}'
                    )

                else:
                    res = pd.read_csv(
                        f'{sdir}'
                        '/adaptive'
                        '/general_bias'
                        '/inter_x_intra_species_bias_adaptive.csv'
                    )

            if 'transition' in file:
                mst = pd.read_csv(
                    f'{sdir}'
                    '/de_novo'
                    '/transition_bias'
                    '/transition_bias_de_novo.csv'
                )

                ast = pd.read_csv(
                    f'{sdir}'
                    '/adaptive'
                    '/transition_bias'
                    '/transition_bias_adaptive.csv'
                )

                sts = merge(sts, mst[:-3])
                sts = merge(sts, ast[:-3])

            else:
                for m in mutations:
                    mst = pd.read_csv(
                        f'{sdir}'
                        '/de_novo'
                        '/general_bias'
                        f'/{m}_general_bias_de_novo.csv'
                    )

                    ast = pd.read_csv(
                        f'{sdir}'
                        '/adaptive'
                        '/general_bias'
                        f'/{m}_general_bias_adaptive.csv'
                    )

                    sts = merge(sts, mst[:-3])
                    sts = merge(sts, ast[:-3])

            axmin = np.log10(np.min(list(sts['-95%'])))
            axmax = np.log10(np.max(list(sts['+95%'])))
            margin = 0.125 * (axmax - axmin)
            axmin -= margin
            axmax += margin
            linrg = 10.0 ** np.array([axmin, axmax])
            fig, ax = plt.subplots(figsize=(10, 10))
            slps = num_form([min(list(slopes)), max(list(slopes))])

            if abs(float(slps[0])) < 0.001:
                lo = '0.0'
            else:
                lo = slps[0]

            ax.text(
                0.98, 0.02,
                f"Effect ($\u03B2$): {lo}\u2013{slps[1]}",
                color='black',
                fontsize=28,
                transform=ax.transAxes,
                va='bottom',
                ha='right',
                zorder=20
            )

            for i in range(len(slopes)):

                reg_i = 10 ** (
                        (np.log10(linrg) * slopes[i]) + intercepts[i]
                )

                ax.plot(
                    linrg,
                    reg_i,
                    linewidth=4,
                    color=rclr,
                    alpha=1 / 4
                )

            slpi = [i for i in range(len(res))
                    if 'slope' in list(res['var'])[i]][0]

            icpi = [i for i in range(len(res))
                    if 'intercept' in list(res['var'])[i]][0]

            slp, itcp = list(res['val'])[slpi], list(res['val'])[icpi]

            ax.plot(
                linrg,
                10 ** ((np.log10(linrg) * slp) + itcp),
                linewidth=3,
                color='black',
                zorder=5
            )

            if file[:3] in mutations:

                ax.text(
                    0.02, 0.98,
                    f"{file[0]}:{bp[file[0]]}>{file[2]}:{bp[file[2]]}",
                    color=rclr,
                    transform=ax.transAxes,
                    fontsize=24,
                    ha='left',
                    va='top'
                )

            ax.set_xlabel(
                'Mutation bias',
                fontsize=28,
                labelpad=16
            )
            ax.set_ylabel(
                abias_lbl,
                fontsize=28,
                labelpad=16
            )

            ax.set_xlim(10 ** axmin, 10 ** axmax)
            ax.set_ylim(10 ** axmin, 10 ** axmax)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))

            # Formatter: force plain decimal notation
            def tick_label_format(val, _):
                return f"{val:g}"

            ax.xaxis.set_major_formatter(FuncFormatter(tick_label_format))
            ax.yaxis.set_major_formatter(FuncFormatter(tick_label_format))

            for spine in ax.spines.values():
                spine.set_linewidth(1)

            ax.tick_params(
                which='both',
                width=1,
                top=False,
                right=False
            )

            ax.tick_params(
                which='minor',
                length=7.5,
                top=False,
                right=False
            )

            ax.tick_params(
                axis='both',
                which='major',
                length=15,
                top=False,
                right=False,
                labelsize=24
            )

            plt.savefig(
                f'{fdir}'
                '/sans_species'
                f'/{file[:-4]}.pdf',
                bbox_inches='tight'
            )

            plt.close('all')


# Conduct analysis to determine if regression
# predicts adaptive substitution bias for held-out species;
# statistical test is done using permutations of
def fig_predict_held_out(
        rdir, txfig, a_bias_label,
        sspdir, sp_mutn, sim_n
):

    fig, ax = txfig
    sspr = pd.read_csv(f'{sspdir}/all_snm_bias_adaptive.csv')
    a_prd = np.empty((0, 6), int)
    a_emp = np.empty((0, 6), int)

    # beta, and beta_0, to lists for downstream regression analyses:
    # Append predicted & empirical adaptive substitution bias,
    for sp in list(sp_mutn['species']):

        sspred = sspr.loc[sspr['dataset'] == f'sans_{sp}']
        slp = float(sspred['slope'].iloc[0])
        itc = float(sspred['intercept'].iloc[0])
        spdat = pd.read_csv(
            f'{rdir}'
            '/by_species'
            f'/{sp}_outcomes_model_vs_empirical_all_data.csv'
        )

        mb = np.array(spdat['m_bias'])
        a_emp = np.append(a_emp, np.array([list(spdat['a_bias'])]), axis=0)
        a_prd = np.append(
            a_prd,
            np.array([list(10 ** ((np.log10(mb) * slp) + itc))]), axis=0
        )

    l_prd = np.log10(a_prd).flatten()
    l_emp = np.log10(a_emp)

    rperm = np.array([
        np.corrcoef(
            l_prd,
            np.random.permutation(l_emp).flatten())[0][1]
        for i in range(sim_n)
    ])

    l_emp = l_emp.flatten()
    m, b = np.polyfit(l_prd, l_emp, 1)
    r = np.corrcoef(l_prd, l_emp)[0][1]

    pval = sig_simdat(r, rperm, 'upper', 1)

    if pval[0] == '<':
        ptxt = f"p{pval}"
    else:
        ptxt = f"p={pval}"

    axmin = min(min(l_prd), min(l_emp))
    axmax = max(max(l_prd), max(l_emp))
    margin = 0.1 * (axmax - axmin)
    axmin -= margin
    axmax += margin

    ax[0].scatter(
        a_prd.flatten(),
        a_emp.flatten(),
        s=20,
        color='black',
        alpha=0.5
    )

    ax[0].plot(
        10 ** np.array([axmin, axmax]),
        10 ** ((np.array([axmin, axmax]) * m) + b),
        linewidth=1,
        color='black'
    )

    ax[0].text(
        0.02, 0.98,
        f'r={round(r, 2)}',
        fontsize=10,
        transform=ax[0].transAxes,
        ha='left',
        va='top'
    )

    ax[0].text(
        0.02, 0.92,
        ptxt,
        fontsize=10,
        transform=ax[0].transAxes,
        ha='left',
        va='top'
    )

    ax[0].set_xlabel(
        f'{a_bias_label} \n (Predicted)',
        fontsize=11,
        labelpad=4
    )
    ax[0].set_ylabel(
        f'{a_bias_label} \n (Observed)',
        fontsize=11,
        labelpad=4
    )

    ax[0].set_xlim(10 ** axmin, 10 ** axmax)
    ax[0].set_ylim(10 ** axmin, 10 ** axmax)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax[0].yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))

    # Formatter: force plain decimal notation
    def tick_label_format(val, _):
        return f"{val:g}"

    ax[0].xaxis.set_major_formatter(FuncFormatter(tick_label_format))
    ax[0].yaxis.set_major_formatter(FuncFormatter(tick_label_format))

    for spine in ax[0].spines.values():
        spine.set_linewidth(0.5)

    ax[0].tick_params(
        which='both',
        width=0.5,
        top=False,
        right=False
    )

    ax[0].tick_params(
        which='minor',
        length=3,
        top=False,
        right=False
    )

    ax[0].tick_params(
        axis='both',
        which='major',
        length=6,
        top=False,
        right=False,
        labelsize=10
    )

    ax[0].set_box_aspect(1)

    ax[0].text(
        -0.4, 1.1, 'A',
        transform=ax[0].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top"
    )

    return fig, ax


# Graph the summarized results, slope (beta),
# for the regressions that result from
# leaving each species out one at a time:
def fig_sans_species_effects(
        rdir, txfig, nonind, ind,
        mutations, bp, col, colors
):

    fig, ax = txfig
    if nonind == 'sans_one_species':
        ax_i = 1
    elif nonind == 'phylogenetic_nonindependence':
        ax_i = (0, 0)
    else:
        raise RuntimeError(
            "Axes index outside the bounds for plot "
            "of mutation bias effects by regression analysis"
        )

    fx = pd.DataFrame(columns=col)

    ti = pd.read_csv(
        f'{rdir}/{nonind}'
        '/transition_bias_adaptive.csv'
    ).sort_values(
        by='slope',
        ignore_index=True
    )['slope']

    ii = pd.read_csv(
        f'{rdir}/{nonind}'
        '/all_snm_bias_adaptive.csv'
    ).sort_values(
        by='slope',
        ignore_index=True
    )['slope']

    bias = [
        ['Transition' for s in range(len(ind))],
        ['Intra- & interspecies' for s in range(len(ind))]
    ]

    blk = ['black' for s in range(len(ind))]

    fx = merge(
        fx,
        pd.DataFrame(
            zip(bias[0], list(ti), blk),
            columns=col
        )
    )

    fx = merge(
        fx,
        pd.DataFrame(
            zip(bias[1], list(ii), blk),
            columns=col
        )
    )

    clrs = ['black', 'black'] + [colors[m] for m in mutations]

    for m in mutations:

        clr = [colors[m] for s in range(len(ind))]
        ml = [f'{m[0]}:{bp[m[0]]}>{m[2]}:{bp[m[2]]}'
              for s in range(len(ind))]

        mi = pd.read_csv(
            f'{rdir}/{nonind}/{m}_bias_adaptive.csv'
        ).sort_values(
            by='slope',
            ignore_index=True
        )['slope']

        fx = merge(
            fx,
            pd.DataFrame(
                zip(ml, list(mi), clr),
                columns=col
            )
        )

    sns.stripplot(
        data=fx,
        x="Regression",
        y="Effect ($\u03B2$)",
        size=8,
        hue="Regression",
        palette=clrs,
        alpha=0.5,
        ax=ax[ax_i]
    )

    ax[ax_i].plot(
        np.array([0, 7]),
        np.zeros(2),
        linewidth=1,
        linestyle=':',
        color='black'
    )

    ax[ax_i].set_ybound(lower=-0.5, upper=1.5)

    ax[ax_i].set_yticks(
        [-0.5, 0, 0.5, 1.0, 1.5],
        ['-0.5', '0', '0.5', '1', '1.5'],
        fontsize=10
    )

    ax[ax_i].tick_params(
        which='both',
        width=0.5,
        top=False,
        right=False
    )

    for spine in ax[ax_i].spines.values():
        spine.set_linewidth(0.5)

    ax[ax_i].set_xlabel(xlabel="Regression", fontsize=11)
    ax[ax_i].set_ylabel(ylabel="Effect ($\u03B2$)", fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=10)

    if nonind == 'sans_one_species':
        ax[ax_i].text(
            -0.15, 1.1, 'B', transform=ax[ax_i].transAxes,
            fontsize=14, fontweight="bold", va="top"
        )
        return fig, ax
    elif nonind == 'phylogenetic_nonindependence':
        plt.savefig(
            f'{rdir}/figures/effects_per_regression_{nonind}.pdf',
            bbox_inches='tight'
        )

        plt.close('all')


# Generate figure with panels A-B covering leave-one-species-out analyses:
def fig_taxonomic_exclusion_results(
        rdir, abias_lbl, taxon_ind,
        bp, mutations, colors,
        sans, sp_mutn, sim_n
):

    txfig = plt.subplots(
        1, 2, figsize=(10, 4),
        gridspec_kw={'width_ratios': [1, 2]},
        layout='compressed'
    )

    fig_predict_held_out(
        rdir, txfig, abias_lbl, f'{rdir}/{sans}', sp_mutn, sim_n)

    fig_sans_species_effects(
        rdir, txfig, sans, taxon_ind, mutations, bp,
        ['Regression', 'Effect ($\u03B2$)', 'color'], colors
    )

    plt.savefig(f'{rdir}/figures/robustness_to_taxonomic_exclusion.pdf')
    plt.close('all')


# Generate figure with panels A-C showing the results from the reanalysis
# of the 3 main regressions (transition bias, inter-by-intra species bias,
# the combined regression of six mutation class-specific biases):
def fig_reanalysis_of_main_results(
        rdir, model, spectrum, abias_lbl,
        mutations, bp, colors, ucdsym
):

    sfig = plt.subplots(
        3, 1, figsize=(3 + 1/3, 12 + 2/9),
        gridspec_kw={'hspace': 0.2},
        squeeze=False
    )

    fig_combined_regressions(
        rdir, sfig, abias_lbl,
        f'{rdir}/analysis/{model}/statistics',
        mutations, model, spectrum, colors,
        markers=False, reanalysis=True
    )

    fig_xspecies_regressions(
        f'{rdir}/analysis/{model}/statistics'
        f'/{spectrum}/transition_bias',
        sfig, abias_lbl, bp, model, spectrum,
        colors, ucdsym, reanalysis=True
    )

    fig_inter_x_intra_species(
        rdir, sfig, abias_lbl, mutations,
        model, 'general', spectrum, reanalysis=True
    )

    plt.savefig(
        f'{rdir}/figures/{model}/main_analyses_{spectrum}.pdf',
        bbox_inches='tight'
    )

    plt.close('all')


# Generate output figures of spectra and regressions on the
# mutation and adaptive data:
def figures(
        rdir, measure, mutations, bp, sim_n,
        sp_mutn, btypes, data, spectra, fd, colors,
        ucdsps, ucdsym, taxon_ind, phylo_ind
):

    if measure == 'rates':
        abias_lbl = 'Adaptive substitution bias'
    else:
        abias_lbl = 'Bias in adaptive outcome'

    if data == 'all_data':

        if not os.path.exists(f'{rdir}/figures'):
            os.makedirs(f'{rdir}/figures')

        if len(list(sp_mutn['species'])) <= 15:

            fig_intra_species_regr(
                data, abias_lbl, f'{rdir}/analysis',
                f'{rdir}/by_species', f'{rdir}/figures',
                mutations, bp, sim_n, sp_mutn, colors, ucdsps, fd
            )

            print("Intraspecific variation regressions plotted")

        fig_sans_species(
            rdir, abias_lbl, f'{rdir}/sans_one_species',
            f'{rdir}/figures', mutations, bp, colors
        )

        fig_sans_species_effects(
            rdir,
            plt.subplots(
                figsize=(6 + 2 / 3, 4),
                layout='compressed', squeeze=False
            ),
            'phylogenetic_nonindependence', phylo_ind, mutations, bp,
            ['Regression', 'Effect ($\u03B2$)', 'color'], colors
        )

        fig_taxonomic_exclusion_results(
            rdir, abias_lbl, taxon_ind, bp, mutations,
            colors, 'sans_one_species', sp_mutn, sim_n
        )

        print("Taxonomic exclusion analyses plotted")

        fig_bar_graphs(
            rdir, f'{rdir}/by_species', mutations,
            bp, measure, data, colors)

    for spectrum in spectra:

        if data == 'all_data':

            fig_xspecies_regressions(
                f'{rdir}/analysis/model/statistics/{spectrum}',
                plt.subplots(
                    figsize=(3 + 1 / 3, 3 + 1 / 3), squeeze=False),
                abias_lbl, bp, 'model', spectrum,
                colors, ucdsym, reanalysis=False
            )

            print("Regressions plotted for interspecific variation")

            if measure == 'freqs':

                fig_reanalysis_of_main_results(
                    rdir, 'model', spectrum, abias_lbl,
                    mutations, bp, colors, ucdsym
                )
                
                print("Reanalysis of main results plotted")

            elif measure == 'rates':

                fig_combined_regressions(
                    rdir,
                    plt.subplots(
                        figsize=(3 + 1 / 3, 3 + 1 / 3), squeeze=False),
                    abias_lbl, f'{rdir}/analysis/model/statistics',
                    mutations, 'model', spectrum, colors,
                    markers=False, reanalysis=False
                )

                print("Combined regressions plotted")

                fig_inter_x_intra_species(
                    rdir,
                    plt.subplots(
                        figsize=(3 + 1 / 3, 3 + 1 / 3), squeeze=False),
                    abias_lbl, mutations, 'model', 'general', spectrum,
                    reanalysis=False
                )

                print("Inter- by intraspecific regressions plotted")

            for biastype in btypes[2:]:

                fig_multi_regressions(
                    f'{rdir}/analysis', abias_lbl, mutations, bp,
                    len(list(sp_mutn['species'])), 'model', biastype,
                    spectrum, colors, ucdsym
                )

                print("Multi-plot regressions plotted")

            if 'selection_pseudoreplicates' in rdir or 'amino' in rdir:

                fig_reanalysis_of_main_results(
                    rdir, 'model', spectrum, abias_lbl,
                    mutations, bp, colors, ucdsym
                )

                print("Reanalysis of main results plotted")

    if measure == 'rates' and data == 'all_data':

        fig_combined_regressions(
            rdir,
            plt.subplots(figsize=(3 + 1 / 3, 3 + 1 / 3), squeeze=False),
            abias_lbl, f'{rdir}/analysis/model/statistics',
            mutations, 'model', 'adaptive', colors,
            markers=True, reanalysis=False
        )

        print("Combined regressions plotted for rates bias on all data")

        for weights in ['weighted_adaptive', 'weighted_total']:

            fig_reanalysis_of_main_results(
                rdir, 'model', weights, abias_lbl,
                mutations, bp, colors, ucdsym
            )

            print("Weighted regression results plotted")


# Generate graph summarizing the results from assessing the impact of
# neutral contamination of adaptive data on the mutation-vs-adaptation
# regressions:
def fig_contam_effect(
        measure, mutations, bp,
        sim_n, ac, colors
):

    bias = [
        'transition_bias',
        'inter_x_intra_species_bias',
        f'{mutations[0]}_intersp_panmut_general_bias'
    ]

    bdir = [
        'transition',
        'general',
        'general'
    ]

    lbls = [
        'Effect (transition)',
        'Effect (intra-/inter-)',
        'Effect (interspecies)'
    ]

    cols = [
        'effect',
        'errmid',
        'err', 'p'
    ]

    sttl = [
        pd.DataFrame(columns=cols),
        pd.DataFrame(columns=cols),
        pd.DataFrame(columns=cols)
    ]

    stmi = [pd.DataFrame(columns=cols) for m in mutations]

    trmd = [
        s for s in os.listdir(
            './results'
            f'/results_with_stats_on_{sim_n}_sims'
            f'/bias_in_{measure}'
        )
        if 'trimmed_data' in s
    ]

    trmd.sort()

    resd = f'./results/results_with_stats_on_{sim_n}_sims'
    mrdr = f'{resd}/bias_in_{measure}'

    for subd in ['all_data'] + trmd:

        sdir = f'{mrdr}/{subd}/analysis/model/statistics/adaptive'
        nuld = f'{mrdr}/{subd}/analysis/null/bootstrap_results/adaptive'

        for bi in bias:

            iso = pd.read_csv(
                f'{sdir}'
                f'/{bdir[bias.index(bi)]}_bias'
                f'/{bi}_adaptive.csv'
            )

            nul = np.array(
                list(
                    pd.read_csv(
                        f'{nuld}'
                        f'/{bdir[bias.index(bi)]}_bias'
                        f'/{bi}_adaptive.csv'
                    )
                    ['slope']
                )
            )[1:]

            sttl[bias.index(bi)] = contam_sumstats(
                iso, nul, 'slope',
                sttl[bias.index(bi)], cols
            )

        for m in mutations:

            iso = pd.read_csv(
                f'{sdir}'
                '/general_bias'
                f'/{m}_general_bias_adaptive.csv'
            )

            nul = np.array(
                list(
                    pd.read_csv(
                        f'{nuld}'
                        '/general_bias'
                        f'/{m}_general_bias_adaptive.csv'
                    )
                    ['slope']
                )
            )[1:]

            stmi[mutations.index(m)] = contam_sumstats(
                iso, nul, 'slope',
                stmi[mutations.index(m)], cols
            )

    fig, axs = plt.subplots(4, 2, figsize=(10, 10))
    cr = np.array([0] + ac)
    xspc = ac[-1] / 4

    mlbl = {
        'A>C': [xspc, 2.16],
        'A>G': [0, 2.16],
        'A>T': [xspc, 1.82],
        'G>A': [0, 1.82],
        'G>C': [xspc * 2, 2.16],
        'G>T': [xspc * 2, 1.82]
    }

    for r in range(0, 3):

        axs[r, 0].plot(
            cr,
            list(sttl[r]['effect']),
            linewidth=2,
            color='black'
        )

        axs[r, 0].scatter(
            cr,
            list(sttl[r]['effect']),
            s=25,
            color='black'
        )

        axs[r, 0].errorbar(
            cr,
            list(sttl[r]['errmid']),
            yerr=list(sttl[r]['err']),
            fmt='none',
            capsize=2,
            capthick=1,
            linewidth=1,
            color='black'
        )

        axs[r, 0].set_yticks(
            [-1, 0, 1, 2],
            ['-1', '0', '1', '2'],
            fontsize=10
        )

        axs[r, 0].set_ylabel(
            lbls[r],
            fontsize=12,
            labelpad=12
        )

        axs[r, 1].plot(
            cr,
            list(sttl[r]['p']),
            linewidth=2,
            color='black'
        )

        axs[r, 1].scatter(
            cr,
            list(sttl[r]['p']),
            s=25,
            color='black'
        )

        axs[r, 1].set_ylabel(
            f'p ({lbls[r][8:-1]})',
            fontsize=12,
            labelpad=12
        )

    for mi in range(0, 6):

        m = mutations[mi]

        axs[3, 0].plot(
            cr,
            list(stmi[mi]['effect']),
            linewidth=2,
            color=colors[m],
            alpha=0.5
        )

        axs[3, 0].errorbar(
            cr,
            list(stmi[mi]['errmid']),
            yerr=list(stmi[mi]['err']),
            fmt='none',
            capsize=2,
            capthick=1,
            linewidth=1,
            color=colors[m],
            alpha=0.5
        )

        axs[3, 0].text(
            mlbl[m][0],
            mlbl[m][1],
            f'{m[0]}:{bp[m[0]]}>{m[2]}:{bp[m[2]]}',
            color=colors[m],
            fontsize=12
        )

        axs[3, 1].plot(
            cr,
            list(stmi[mi]['p']),
            linewidth=2,
            color=colors[mutations[mi]],
            alpha=0.5
        )

    for r in range(0, 4):

        for rp in range(0, 2):

            axs[r, rp].set_xticks(
                cr,
                ['0'] + [str(round(c, 1)) for c in ac],
                fontsize=10
            )

        axs[r, 0].plot(
            cr,
            np.zeros(len(ac) + 1),
            linewidth=1,
            linestyle=':',
            color='black'
        )

        axs[r, 1].plot(
            cr,
            np.ones(len(ac) + 1) * 0.05,
            linewidth=1,
            linestyle=':',
            color='black'
        )

        axs[r, 1].text(
            0.0, 0.06,
            '(p=0.05)',
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize=10
        )

        axs[r, 1].set_yscale('log')

        axs[r, 1].set_ybound(lower=0.00005, upper=1)

        axs[r, 1].set_yticks(
            [0.0001, 0.001, 0.01, 0.1, 1],
            ['0.0001', '0.001', '0.01', '0.1', '1'],
            fontsize=10
        )

        axs[r, 0].set_ybound(lower=-1, upper=2)

    axs[3, 0].set_ybound(lower=-1, upper=2.5)

    axs[3, 0].set_yticks(
        [-1, 0, 1, 2],
        ['-1', '0', '1', '2'],
        fontsize=10
    )

    axs[3, 0].set_ylabel(
        'Effect (per SNM)',
        fontsize=12,
        labelpad=12
    )

    axs[3, 1].set_ylabel(
        'p (per SNM)',
        fontsize=12,
        labelpad=12
    )

    for rp in range(0, 2):

        axs[3, rp].set_xlabel(
            'Fraction attributed to contamination',
            fontsize=12,
            labelpad=12
        )

    plt.tight_layout()

    plt.savefig(
        './results'
        f'/results_with_stats_on_{sim_n}_sims'
        f'/bias_in_{measure}'
        '/effects_non_adaptive_contamination.pdf'
    )

    plt.close('all')
