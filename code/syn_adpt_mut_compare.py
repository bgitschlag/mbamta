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
rdir = f'./results/results_with_stats_on_{sim_n}_sims'
mut_n = pd.read_csv(f'./data/species_mutation_spectrum_counts.csv')

bs = ['A', 'C', 'G', 'T']
bp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}


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


# Dictionary specifying the standard genetic code
code = {
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


# Generate spectrum of available synonymous mutation paths,
# to obtain the synonymous-specific substitutions spectra:
def syn_paths(muts, code, bp, bs, codon_use, path_spec, spectrum):

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

                    # Make sure the mutation preserves the amino acid:
                    if code[codon] == code[mut]:

                        ref, alt, path = codon[position], alt_base, 1

                        # Confirm that reference codon is set to the
                        # purine component of the reference base-pair:
                        if codon[position] in [bs[1], bs[3]]:
                            ref, alt = bp[ref], bp[alt]

                        # Codon contributions to nominal and
                        # weighted spectra of mutational paths:
                        pths_frm_cdn[muts.index(f'{ref}>{alt}')] += path

            # Take product of mutation paths from codon &
            # codon-use fraction & add to an array:
            path_spec += cfreq * pths_frm_cdn

    # Normalize and add as columns to table of path spectra:
    # spectrum['code_freq'] = counts / sum(counts)
    spectrum['paths'] = path_spec / sum(path_spec)

    return spectrum


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
        error = (hi - lo) / 2
        coordinates.append([lo, hi, lo + error, error])
        del boot, lo, hi, error

    return coordinates


def plot_intraspecific_regression(
        ax, col, species, sp, xbias, ybias,
        xcoord, ycoord, boot, null_slopes,
        xlabel, ylabel
):

    # Fit regression with bootstrap summary stats:
    m, b = np.polyfit(np.log10(xbias), np.log10(ybias), 1)
    btrgr = [
        np.polyfit(np.log10(xbias), np.log10(boot[r][0]), 1)
        for r in range(sim_n)
    ]
    btslp = np.sort([i[0] for i in btrgr])
    loslp = round(btslp[round(0.025 * len(btslp))], 2)
    hislp = round(btslp[round(0.975 * len(btslp)) - 1], 2)
    p = np.sum(null_slopes >= m) / sim_n
    if p == 0:
        ptxt = f"p<{1 / sim_n}"
    else:
        ptxt = f"p={p}"


    # Figure axis formatting based on data range:
    axmin = np.log10(min(
        min([xcoord[mi][0] for mi in range(6)]),
        min([ycoord[mi][0] for mi in range(6)])
    ))
    axmax = np.log10(max(
        max([xcoord[mi][1] for mi in range(6)]),
        max([ycoord[mi][1] for mi in range(6)])
    ))
    margin = 0.15 * (axmax - axmin)
    axmin -= margin
    axmax += margin
    linrg = 10 ** np.array([axmin, axmax])

    # Create figure legend:
    y_pos = np.linspace(1.8, 1.2, 6)
    for i, y in enumerate(y_pos):

        ax[0, 0].text(
            0.2, y,
            f'{ucdsps[i]}',
            transform=ax[0, 0].transAxes,
            color=colors[mutations[i]],
            fontsize=14,
            ha='center',
            va='center'
        )

        ax[0, 0].text(
            0.3, y,
            f'{mutations[i][0]}:{bp[mutations[i][0]]}'
            f'>{mutations[i][-1]}:{bp[mutations[i][-1]]}',
            transform=ax[0, 0].transAxes,
            color=colors[mutations[i]],
            fontsize=14,
            ha='left',
            va='center'
        )

    # Plot datapoints with error bars:
    for mi in range(6):

        ax[species.index(sp), col].text(
            xbias[mi], ybias[mi], ucdsps[mi],
            color=colors[mutations[mi]],
            fontsize=14,
            horizontalalignment='center',
            verticalalignment='center',
            zorder=20
        )

        ax[species.index(sp), col].errorbar(
            xbias[mi], ycoord[mi][2],
            yerr=ycoord[mi][3],
            color=colors[mutations[mi]],
            linewidth=1,
            capsize=4,
            alpha=0.5,
            zorder=15
        )


        ax[species.index(sp), col].errorbar(
            xcoord[mi][2], ybias[mi],
            xerr=xcoord[mi][3],
            color=colors[mutations[mi]],
            linewidth=1,
            capsize=4,
            alpha=0.5,
            zorder=15
        )


    # Plot empirical and bootstrap regressions:
    ax[species.index(sp), col].plot(
        linrg,
        10 ** (m * np.log10(linrg) + b),
        color='black',
        linewidth=1,
        alpha=0.5,
        zorder=10
    )

    for i in range(sim_n):
        ax[species.index(sp), col].plot(
            linrg,
            10 ** (btrgr[i][0] * np.log10(linrg) + btrgr[i][1]),
            linewidth=0.5,
            color='xkcd:silver',
            alpha=1 / 3,
            zorder=0
        )


    # Format axes; plot species & summary statistics:
    ax[species.index(sp), col].set(aspect='equal', zorder=10)
    ax[species.index(sp), col].set_xlim(10 ** axmin, 10 ** axmax)
    ax[species.index(sp), col].set_ylim(10 ** axmin, 10 ** axmax)
    ax[species.index(sp), col].set_xscale('log')
    ax[species.index(sp), col].set_yscale('log')
    ax[species.index(sp), col].xaxis.set_major_locator(
        LogLocator(base=10, subs=(1.0,)))
    ax[species.index(sp), col].yaxis.set_major_locator(
        LogLocator(base=10, subs=(1.0,)))

    ax[species.index(sp), col].text(
        0.025, 0.975,
        sp[0].upper() + '. ' + sp[2:],
        fontsize=10,
        fontstyle='italic',
        transform=ax[species.index(sp), col].transAxes,
        ha='left',
        va='top'
    )

    ax[species.index(sp), col].text(
        0.975, 0.025,
        f"$\u03B2$={round(m, 2)} ({loslp}\u2013{hislp})",
        fontsize=10,
        transform=ax[species.index(sp), col].transAxes,
        ha='right',
        va='bottom',
        zorder=20
    )

    ax[species.index(sp), col].text(
        0.975, 0.125,
        ptxt,
        fontsize=10,
        transform=ax[species.index(sp), col].transAxes,
        ha='right',
        va='bottom',
        zorder=20
    )


    # Format axes: force plain decimal notation
    def tick_label_format(val, _):
        return f"{val:g}"

    ax[species.index(sp), col].xaxis.set_major_formatter(
        FuncFormatter(tick_label_format))
    ax[species.index(sp), col].yaxis.set_major_formatter(
        FuncFormatter(tick_label_format))
    ax[species.index(sp), col].xaxis.set_minor_formatter(
        NullFormatter())
    ax[species.index(sp), col].yaxis.set_minor_formatter(
        NullFormatter())

    ax[species.index(sp), col].tick_params(
        axis='both', which='major', length=4,
        top=False, right=False, labelsize=10
    )

    for spine in ax[species.index(sp), col].spines.values():
        spine.set_linewidth(0.5)

    ax[species.index(sp), col].set_xlabel(xlabel, fontsize=11)
    ax[species.index(sp), col].set_ylabel(ylabel, fontsize=11)

    return ax


# Perform and plot least squares linear regression analysis
# on log-transformed synonymous substitution bias vs mutation bias:
def plot_multiple_intraspecific_regressions(
        species, data, effects,
        changes, mut_n, fig, ax
):

    # Plot synonymous bias against mutation bias, with
    # bootstrap uncertainty and statistical significance, using
    # species-specific datasets for which such data are available:
    for sp in species:

        # Call spectra file, mutation & adaptive spectra,
        # bootstraps & statistics:
        spectra = pd.read_csv(
            f'{rdir}/bias_in_rates/all_data/by_species'
            f'/{sp}_outcomes_model_vs_empirical_all_data.csv'
        )

        mbias = np.array(spectra['m_bias'])
        abias = np.array(spectra['a_bias'])
        sp_n = int(mut_n.loc[mut_n['species'] == sp]['mu_counts'].iloc[0])
        mspec = np.array(spectra['mu'])
        mspec = (mspec * (sp_n + 6)) - 1
        mspec = np.clip(mspec, 0, None)
        mspec /= mspec.sum()

        mboot = mba.bootstrap(
            "rates", "de_novo", [mspec], [1], sim_n, [sp_n], [np.ones(6)])
        mboot = mboot / (1 - mboot)
        aboot = np.array(pd.read_csv(
            f'{rdir}/bias_in_rates/all_data/analysis/model'
            f'/bootstrap_results/adaptive/within_species'
            f'/{sp}_intra_sp_bias_adaptive.csv'
        ).iloc[1:, 1:-3]).reshape(sim_n, 1, 6)
        mcoord = bootstats(mboot.T)
        acoord = bootstats(aboot.T)


        # Generate spectra of synonymous rates bias
        # with bootstrap statistics:
        sp_chg = pd.read_csv(f'./data/adaptive_changes/{data[sp]}.csv')
        sp_syn = sp_chg.loc[sp_chg['Effect'].str.contains(
            effects[sp], regex=True, na=False), :]
        synspec = np.ones(6)
        for i in range(len(sp_syn)):
            if len(changes[sp]) == 1:
                m = sp_syn[changes[sp][0]].iloc[i]
                ref, alt = m[0], m[-1]
            else:
                if len(sp_syn[changes[sp][1]].iloc[i]) == 1:
                    ref = sp_syn[changes[sp][0]].iloc[i]
                    alt = sp_syn[changes[sp][1]].iloc[i]
                else:
                    continue
            synspec[mutations.index(standard_mut(f'{ref}>{alt}'))] += 1


        syn_n = int(sum(synspec) - 6)
        paths = np.array(syn_paths(
            mutations, code, bp, bs, mba.codon_use(sp), np.zeros(6),
            pd.DataFrame(zip(mutations), columns=['mutation'])
        )['paths'])
        sboot = mba.bootstrap(
            "rates", "adaptive", [synspec - 1],
            [1], sim_n, [syn_n], [paths]
        )
        synspec /= paths
        synspec = synspec / sum(synspec)
        sbias = synspec / (1 - synspec)
        sboot = sboot / (1 - sboot)
        scoord = bootstats(sboot.T)


        # Simulated regressions under null model:
        nboot_s = mba.bootstrap(
            "rates", "adaptive", [paths], [1], sim_n, [syn_n], [paths])
        nboot_s = nboot_s / (1 - nboot_s)

        nboot_a = np.array(
            pd.read_csv(
                f'{rdir}/bias_in_rates/all_data/analysis'
                '/null/bootstrap_results/adaptive/within_species'
                f'/{sp}_intra_sp_bias_adaptive.csv',
                usecols=[m for m in mutations]
            )[1:]
        )

        nboot_a_slp = np.array(
            pd.read_csv(
                f'{rdir}/bias_in_rates/all_data/analysis'
                '/null/bootstrap_results/adaptive/within_species'
                f'/{sp}_intra_sp_bias_adaptive.csv',
                usecols=['slope']
            )[1:]
        )

        nslp_s_v_m = [i[0] for i in [
            np.polyfit(np.log10(mbias), np.log10(nboot_s[r][0]), 1)
            for r in range(sim_n)
        ]]

        nslp_a_v_s = [i[0] for i in [
            np.polyfit(np.log10(sbias), np.log10(nboot_a[r]), 1)
            for r in range(sim_n)
        ]]

        ax = plot_intraspecific_regression(
            ax, 0, species, sp, mbias, abias, mcoord, acoord,
            aboot, nboot_a_slp, "Mutation bias",
            "Adaptive \n substitution bias"
        )
        ax = plot_intraspecific_regression(
            ax, 1, species, sp, mbias, sbias, mcoord, scoord,
            sboot, nslp_s_v_m, "Mutation bias",
            "Synonymous \n substitution bias"
        )
        ax = plot_intraspecific_regression(
            ax, 2, species, sp, sbias, abias, scoord, acoord,
            aboot, nslp_a_v_s, "Synonymous \n substitution bias",
            "Adaptive \n substitution bias"
        )

    return fig, ax


fig, ax = plt.subplots(3, 3, figsize=(10, 10), gridspec_kw={'wspace': 0.5})
plot_multiple_intraspecific_regressions(
    ['e_coli', 'p_falciparum', 's_cerevisiae'],
    args['syn_data'], args['syn_effects'], args['syn_changes'],
    mut_n, fig, ax
)

# Label axes and save figure to output folder:
plt.savefig(
    './results'
    f'/results_with_stats_on_{sim_n}_sims'
    '/bias_in_rates'
    '/all_data'
    '/figures/synonymous_vs_mutation_bias.pdf',
    bbox_inches='tight'
)
plt.close()