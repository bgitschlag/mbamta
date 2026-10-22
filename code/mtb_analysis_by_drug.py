# Import packages that will be needed for this script:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
import mbamta_analyses as mba


# Open configuration file to execute input arguments:
args = {}
with open(f'./mbamta_config.txt') as f:
    exec(f.read(), {}, args)

# Number of datasets to simulate:
sim_n = int(args['simulations'])

basel = pd.read_csv(args['basel'])
manson = pd.read_csv(args['manson'])
mut_n = pd.read_csv(f'./data/species_mutation_spectrum_counts.csv')

mtb_spectra = pd.read_csv(
    f'./results/results_with_stats_on_{sim_n}_sims'
    '/bias_in_rates/all_data/by_species'
    '/m_tuberculosis_outcomes_model_vs_empirical_all_data.csv'
)

bs = ['A', 'C', 'G', 'T']
bp = {
    'A': 'T', 'C': 'G',
    'G': 'C', 'T': 'A'
}


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
    u"\u25CF", u"\u25C6", u"\u25B2",
    u"\u25BC", u"\u25B6", u"\u25C0"
]


# Amino acid abbreviations
amino_acid = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D",
    "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
    "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
    "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"
}


# Compute GC content for an input sequence:
def gc_content(seq):

    nucleotide_n = np.zeros(4)
    with open(seq) as f:
        gene_sequence = f.read()
    for position in range(len(gene_sequence)):
        if gene_sequence[position] in bs:
            nucleotide_n[bs.index(gene_sequence[position])] += 1

    return sum(nucleotide_n[1:3]) / sum(nucleotide_n)


# Standardize the mutation class formatting:
def standard_mut(mut):
    if mut[0] in ['C', 'T']:
        return f'{bp[mut[0]]}>{bp[mut[2]]}'
    else:
        return mut


# Generate summary statistics
# (top, bottom, middle and +/- of 95% bootstrap confidence range):
def bootstats(bootstraps):

    coordinates = []

    for mi in range(6):

        boot = np.sort(bootstraps[mi][0])
        lo = boot[round(0.025 * len(boot))]
        hi = boot[round(0.975 * len(boot)) - 1]
        lo, hi = lo / (1 - lo), hi / (1 - hi)
        error = (hi - lo) / 2
        coordinates.append([lo + error, error])
        del boot, lo, hi, error

    return coordinates


# Perform and plot least squares linear regression analysis
# on log-transformed adaptive substitution bias vs mutation bias
# per selection condition or sequence data type (e.g. "noncoding"):
def mtb_regression_by_drug(
        basel, manson,
        mtb_spectra,
        mut_n, rrs
):

    mtb_n = int(
        mut_n.loc[
            mut_n['species'] == 'm_tuberculosis'
        ]['mu_counts'].iloc[0]
    )

    mbias = np.array(mtb_spectra['m_bias'])
    mspec = np.array(mtb_spectra['mu'])
    mspec = (mspec * (mtb_n + 6)) - 1
    mspec = np.clip(mspec, 0, None)
    mspec /= mspec.sum()

    baseldrugs = {
        "Ethambutol": "EMB",
        "Ethionamide": "ETH",
        "Fluoroquinolone": "FQ",
        "Isoniazid": "INH",
        "Kanamycin": "KAN",
        "Pyrazinamide": "PZA",
        "Rifampicin": "RIF",
        "Streptomycin": "SM"
    }

    drugs = [
        'Ethambutol', 'Pyrazinamide',
        'Fluoroquinolone', 'Streptomycin',
        'Ethionamide', 'Isoniazid',
        'Rifampicin', 'Noncoding'
    ]

    aa_abbr = ''.join(amino_acid.values())
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    ax[0, 2].axis("off")
    row = 0


    # Bootstrap uncertainty in M. tuberculosis mutation spectrum:
    mcoord = bootstats(
        mba.bootstrap(
            "rates", "de_novo",
            [mspec], [1],
            sim_n, [mtb_n],
            [np.ones(6)]
        ).T
    )


    # Plot bias in adaptive outcome against mutation bias,
    # with bootstrap uncertainty and statistical significance,
    # using adaptive outcomes from Basel & Manson datasets:
    for drug in drugs:

        # Specify plot position in figure:
        di = drugs.index(drug)
        if di in [1, 3, 6]:
            col = 1
        elif di in [4, 7]:
            col = 2
        else:
            col = 0
        if di in [2, 5]:
            row += 1
        crd = [row, col]


        # Get adaptive substitutions spectra from Basel & Manson data:
        spec = np.ones(6)
        if drug == 'Noncoding':
            gc_nc = gc_content(rrs)
            at_nc = 1 - gc_nc
            paths = np.concatenate((
                np.ones(3) * at_nc,
                np.ones(3) * gc_nc
            ))
            paths = paths / sum(paths)
            drugbs = basel.loc[basel['LOCUS'] == 'rrs', :]
            drugms = manson.loc[manson['Gene'] == 'rrs', :]

        else:
            paths = np.array(mtb_spectra['paths'])
            baseldrug = baseldrugs[drug]
            drugbs = basel.loc[basel['DRUG'] == baseldrug, :]
            drugbs = drugbs.loc[
                     drugbs['SUBSTITUTION'].str.match(
                         rf'^([{aa_abbr}])\d+(?!\1$)[{aa_abbr}]$',
                         na=False
                     ), :]
            if drug == 'Fluoroquinolone':
                mansondrug = 'Ofloxacin'
            else:
                mansondrug = drug
            drugms = manson.loc[manson['Drug(s)'] == mansondrug, :]
            drugms = drugms.loc[
                     drugms['Amino Acid Change'].str.match(
                         rf'^([{aa_abbr}])\d+(?!\1$)[{aa_abbr}]$',
                         na=False
                     ), :]

        drugms = drugms.loc[
                 drugms[
                     'mutation in genomic coordinates'
                 ].str.contains(
                     r':[ACGT]->[ACGT]$'
                 ), :]

        for i in range(len(drugbs)):
            ref = drugbs['REF_BASE'].iloc[i]
            alt = drugbs['ALT_BASE'].iloc[i]
            spec[
                mutations.index(standard_mut(f'{ref}>{alt}'))
            ] += drugbs['EVENTS'].iloc[i]
        for i in range(len(drugms)):
            m = drugms['mutation in genomic coordinates'].iloc[i]
            spec[
                mutations.index(standard_mut(f'{m[-4]}>{m[-1]}'))
            ] += drugms['# independent evolutions'].iloc[i]


        # Get spectra of adaptive rates bias, with bootstrap samples:
        adpn = int(sum(spec) - 6)
        aboot = mba.bootstrap(
            "rates", "adaptive", [spec - 1],
            [1], sim_n, [adpn], [paths]
        )
        acoord = bootstats(aboot.T)
        spec /= paths
        spec = spec / sum(spec)
        abias = spec / (1 - spec)
        aboot = aboot / (1 - aboot)


        # Condition-specific regression with bootstrap confidence:
        m, b = np.polyfit(np.log10(mbias), np.log10(abias), 1)
        btrgr = [
            np.polyfit(
                np.log10(mbias),
                np.log10(aboot[r][0]),
                1
            ) for r in range(sim_n)
        ]
        btslp = np.sort([i[0] for i in btrgr])
        loslp = round(btslp[round(0.025 * len(btslp))], 2)
        hislp = round(btslp[round(0.975 * len(btslp)) - 1], 2)

        # Figure axis formatting based on data range:
        all_vals = []
        for mi in range(6):
            all_vals.append(mcoord[mi][0] - mcoord[mi][1])
            all_vals.append(mcoord[mi][0] + mcoord[mi][1])
            all_vals.append(acoord[mi][0] - acoord[mi][1])
            all_vals.append(acoord[mi][0] + acoord[mi][1])
            all_vals.append(mbias[mi])
            all_vals.append(abias[mi])

        axmin = np.log10(min(all_vals))
        axmax = np.log10(max(all_vals))
        margin = 0.15 * (axmax - axmin)
        axmin -= margin
        axmax += margin
        linrg = 10 ** np.array([axmin, axmax])
        ax[crd[0], crd[1]].set_xlim(linrg)
        ax[crd[0], crd[1]].set_ylim(linrg)


        # Significance of model regression against null model:
        nboot = mba.bootstrap(
            "rates", "adaptive", [paths],
            [1], sim_n, [adpn], [paths]
        )
        nrgr = [
            np.polyfit(
                np.log10(mbias),
                np.log10(nboot[r][0]), 1
            ) for r in range(sim_n)
        ]
        p = np.sum([i[0] for i in nrgr] >= m) / sim_n
        if p == 0:
            ptxt = f"p<{1 / sim_n}"
        else:
            ptxt = f"p={p}"


        # Plot datapoints with error bars:
        for mi in range(6):

            ax[crd[0], crd[1]].text(
                mbias[mi],
                abias[mi],
                ucdsps[mi],
                color=colors[mutations[mi]],
                fontsize=14,
                horizontalalignment='center',
                verticalalignment='center',
                zorder=20
            )

            ax[crd[0], crd[1]].errorbar(
                mbias[mi],
                acoord[mi][0],
                yerr=acoord[mi][1],
                color=colors[mutations[mi]],
                linewidth=1,
                capsize=4,
                alpha=0.5,
                zorder=15
            )

            ax[crd[0], crd[1]].errorbar(
                mcoord[mi][0],
                abias[mi],
                xerr=mcoord[mi][1],
                color=colors[mutations[mi]],
                linewidth=1,
                capsize=4,
                alpha=0.5,
                zorder=15
            )


        # Plot empirical and bootstrap regressions:
        ax[crd[0], crd[1]].plot(
            linrg,
            10 ** (m * np.log10(linrg) + b),
            color='black',
            linewidth=1,
            alpha=0.5,
            zorder=10
        )

        for i in range(sim_n):
            ax[crd[0], crd[1]].plot(
                linrg,
                10 ** (btrgr[i][0] * np.log10(linrg) + btrgr[i][1]),
                linewidth=0.5,
                color='xkcd:silver',
                alpha=1 / 3,
                zorder=0
            )

        # Format axes; plot condition/drug & summary statistics:
        ax[crd[0], crd[1]].set(aspect='equal', zorder=10)
        ax[crd[0], crd[1]].set_xlim(10 ** axmin, 10 ** axmax)
        ax[crd[0], crd[1]].set_ylim(10 ** axmin, 10 ** axmax)
        ax[crd[0], crd[1]].set_xscale('log')
        ax[crd[0], crd[1]].set_yscale('log')
        ax[crd[0], crd[1]].xaxis.set_major_locator(
            LogLocator(base=10, subs=(1.0,)))
        ax[crd[0], crd[1]].yaxis.set_major_locator(
            LogLocator(base=10, subs=(1.0,)))

        if drug == 'Noncoding':
            condition_or_target = "Ribosomal RNA"
        else:
            condition_or_target = drug

        ax[crd[0], crd[1]].text(
            0.025, 0.975,
            condition_or_target,
            fontsize=11,
            transform=ax[crd[0], crd[1]].transAxes,
            ha='left',
            va='top'
        )

        ax[crd[0], crd[1]].text(
            0.975, 0.025,
            f"$\u03B2$={round(m, 2)} ({loslp}\u2013{hislp})",
            fontsize=11,
            transform=ax[crd[0], crd[1]].transAxes,
            ha='right',
            va='bottom',
            zorder=20
        )

        ax[crd[0], crd[1]].text(
            0.975, 0.125,
            ptxt,
            fontsize=11,
            transform=ax[crd[0], crd[1]].transAxes,
            ha='right',
            va='bottom',
            zorder=20
        )

        # Formatter: force plain decimal notation
        def tick_label_format(val, _):
            return f"{val:g}"

        ax[crd[0], crd[1]].xaxis.set_major_formatter(
            FuncFormatter(tick_label_format))
        ax[crd[0], crd[1]].yaxis.set_major_formatter(
            FuncFormatter(tick_label_format))
        ax[crd[0], crd[1]].xaxis.set_minor_formatter(NullFormatter())
        ax[crd[0], crd[1]].yaxis.set_minor_formatter(NullFormatter())

        ax[crd[0], crd[1]].tick_params(
            which='minor', length=3, top=False, right=False)
        ax[crd[0], crd[1]].tick_params(
            axis='both', which='major', length=6,
            top=False, right=False, labelsize=10
        )

        for spine in ax[crd[0], crd[1]].spines.values():
            spine.set_linewidth(0.5)


    # Create figure legend:
    y_pos = np.linspace(0.8, 0.2, 6)
    for i, y in enumerate(y_pos):

        ax[0, 2].text(
            0.25, y,
            f'{ucdsps[i]}',
            color=colors[mutations[i]],
            fontsize=14,
            ha='center',
            va='center'
        )

        ax[0, 2].text(
            0.33, y,
            f'{mutations[i][0]}:{bp[mutations[i][0]]}'
            f'>{mutations[i][-1]}:{bp[mutations[i][-1]]}',
            color=colors[mutations[i]],
            fontsize=14,
            ha='left',
            va='center'
        )


    # Label axes and save figure to output folder:
    fig.supxlabel('Mutation bias', fontsize=14)
    fig.supylabel('Adaptive substitution bias', fontsize=14)
    plt.tight_layout()
    plt.savefig(
        './results'
        f'/results_with_stats_on_{sim_n}_sims'
        '/bias_in_rates'
        '/all_data'
        '/figures'
        '/m_tuberculosis_regressions_by_drug_class.pdf',
        bbox_inches='tight'
    )
    plt.close()

mtb_regression_by_drug(basel, manson, mtb_spectra, mut_n, args['mtb_rrs'])