"""Figures in the style of Moreni et al. (2025) from a run folder.

Usage: python plots.py <run_dir> [warmup_ms]
Saves PNGs in <run_dir>/figures/. Uses whatever the run folder contains:
  spike_times/  -> raster (Fig 1B-D, 5A, 6A), mean rates vs paper (Fig 2A), single-unit rates (Fig 2B),
                   irregularity CV (Fig 2C left, S6), power spectrum of E rates (S5)
  metrics/perturbation_matrix_<state>.csv -> perturbation matrices (Figs 3A, 4A, 5B, 6B; values as in S7, S8, S11, S12)
                   and, if spont is there, comparison matrices (Figs 3C, 4B, 6D)
The first warmup_ms (default 500) are left out of all rate statistics while the network settles.
"""
import ast
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.signal import welch

from network_parameters import net_dict

POPS = net_dict['populations']
SIZES = dict(zip(POPS, net_dict['num_neurons']))
# Paper colour code per cell type (orange darkened so all four stay distinguishable, also for colour-blind readers)
TYPE_COLOR = {'E': '#D1495B', 'PV': '#4C72B0', 'SST': '#2E933C', 'VIP': '#E59D25'}
LAYERS = ['L1', 'L23', 'L4', 'L5', 'L6']
INK, MUTED, GRID = '#222222', '#666666', '#DDDDDD'
# Model bars of Fig 2A, read off the figure (+-0.2 Hz); L1VIP is not shown there
PAPER_RATES = {'L23E': 0.3, 'L23PV': 2.9, 'L23SST': 3.3, 'L23VIP': 7.6, 'L4E': 1.1, 'L4PV': 3.8, 'L4SST': 1.7,
               'L4VIP': 0.8, 'L5E': 2.1, 'L5PV': 4.4, 'L5SST': 3.7, 'L5VIP': 6.1, 'L6E': 0.9, 'L6PV': 4.7,
               'L6SST': 6.3, 'L6VIP': 3.0}
MATRIX_POPS = POPS[1:] # The paper's matrices are 16x16, without L1VIP

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 200, 'savefig.bbox': 'tight'})


def cell_type(pop):
    return next(t for t in ('SST', 'VIP', 'PV', 'E') if pop.endswith(t))


def layer(pop):
    return next(l for l in LAYERS[::-1] if pop.startswith(l))


def label(pop):
    return pop.replace('L23', 'L2/3 ').replace('L1', 'L1 ').replace('L4', 'L4 ').replace('L5', 'L5 ').replace('L6', 'L6 ')


def load_spikes(run_dir):
    """ {population: {neuron index: array of spike times in ms}} """
    spikes = {}
    for pop in POPS:
        with open(os.path.join(run_dir, 'spike_times', f'{pop}.csv')) as f:
            spikes[pop] = {int(i): np.array(ast.literal_eval(t)) for i, t in csv.reader(f)}
    return spikes


def save(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, name))
    plt.close(fig)
    print('saved', os.path.join(out_dir, name))


def raster(spikes, cfg, out_dir):
    """ Whole column, L1 on top, each layer ordered E, PV, SST, VIP (Fig 1B-D) """
    fig, ax = plt.subplots(figsize=(8, 6))
    offset, layer_start, ticks = 0, 0, []
    for pop in POPS[::-1]: # Bottom-up, so L1 ends on top
        t = np.concatenate([ts for ts in spikes[pop].values()] + [np.array([])])
        y = np.concatenate([np.full(len(ts), offset + i) for i, ts in spikes[pop].items()] + [np.array([])])
        ax.scatter(t, y, s=0.4, color=TYPE_COLOR[cell_type(pop)], linewidths=0, rasterized=True)
        offset += SIZES[pop]
        if pop.endswith('E') or pop == 'L1VIP': # Last population of its layer, going upwards
            ticks.append(((layer_start + offset)/2, label(pop).split()[0]))
            ax.axhline(offset, color=GRID, lw=0.8)
            layer_start = offset
    ax.set_yticks([y for y, _ in ticks], [l for _, l in ticks])
    ax.set_ylim(0, offset)
    ax.set_xlim(0, cfg['t_sim'])
    ax.set_xlabel('Time (ms)')
    inp = cfg.get('input', {})
    if inp.get('state', 'spont') != 'spont':
        ax.axvline(inp['t_state'], color=INK, lw=1, ls='--')
        ax.annotate(f"{inp['state']} input on", (inp['t_state'], offset), xytext=(3, 3), textcoords='offset points',
                    color=INK, fontsize=8)
    if inp.get('pert'):
        ax.axvline(inp['t_pert'], color=MUTED, lw=1, ls=':')
        ax.annotate(f"30 pA to {label(inp['pert'])}", (inp['t_pert'], offset), xytext=(3, -10),
                    textcoords='offset points', color=MUTED, fontsize=8)
    ax.legend(handles=[Patch(color=c, label=t) for t, c in TYPE_COLOR.items()], loc='upper right', ncol=4,
              frameon=False, bbox_to_anchor=(1, 1.08), fontsize=8)
    save(fig, out_dir, 'raster.png')


def neuron_rates(spikes, t0, t1):
    """ {population: array of single-neuron rates (Hz) between t0 and t1}, silent neurons included """
    return {pop: np.array([np.sum((t >= t0) & (t < t1)) for t in trains.values()]) / ((t1 - t0)/1000)
            for pop, trains in spikes.items()}


def rate_bars(rates, out_dir):
    """ Mean rate per population, this run vs the paper's model (Fig 2A) """
    pops = [p for p in POPS if p in PAPER_RATES]
    y = np.arange(len(pops))[::-1]
    fig, ax = plt.subplots(figsize=(5, 7))
    for k, pop in enumerate(pops):
        c = TYPE_COLOR[cell_type(pop)]
        ax.barh(y[k] + 0.2, rates[pop].mean(), height=0.38, color=c)
        ax.barh(y[k] - 0.2, PAPER_RATES[pop], height=0.38, color='white', edgecolor=c, hatch='////', lw=0.8)
    ax.set_yticks(y, [label(p) for p in pops])
    ax.set_xlabel('Firing rate (spikes/s)')
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=MUTED, label='This run'),
                       Patch(facecolor='white', edgecolor=MUTED, hatch='////', label='Paper model (Fig 2A)')],
              frameon=False, loc='lower right')
    save(fig, out_dir, 'mean_rates_vs_paper.png')


def rate_boxplot(rates, out_dir):
    """ Single-unit rates, triangle = population mean (Fig 2B) """
    pops = POPS[1:]
    fig, ax = plt.subplots(figsize=(5, 7))
    box = ax.boxplot([rates[p] for p in pops][::-1], vert=False, patch_artist=True, widths=0.6,
                     medianprops={'color': INK}, flierprops={'markersize': 2, 'markeredgecolor': MUTED})
    for patch, pop in zip(box['boxes'], pops[::-1]):
        patch.set_facecolor(TYPE_COLOR[cell_type(pop)])
        patch.set_edgecolor(INK)
        patch.set_linewidth(0.6)
    ax.scatter([rates[p].mean() for p in pops][::-1], np.arange(1, len(pops) + 1), marker='^', color=INK, s=14,
               zorder=3, label='Mean')
    ax.set_yticks(np.arange(1, len(pops) + 1), [label(p) for p in pops][::-1])
    ax.set_xlabel('Single-unit firing rate (spikes/s)')
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower right')
    save(fig, out_dir, 'single_unit_rates.png')


def cv_bars(spikes, t0, t1, out_dir):
    """ Mean CV of inter-spike intervals per population, neurons with at least 3 spikes (Fig 2C left) """
    pops = POPS[1:]
    cvs = {}
    for pop in pops:
        vals = []
        for t in spikes[pop].values():
            isi = np.diff(t[(t >= t0) & (t < t1)])
            if len(isi) >= 2:
                vals.append(isi.std()/isi.mean())
        cvs[pop] = np.mean(vals) if vals else np.nan
    y = np.arange(len(pops))[::-1]
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.barh(y, [cvs[p] for p in pops], height=0.7, color=[TYPE_COLOR[cell_type(p)] for p in pops])
    ax.axvline(0.5, color=MUTED, lw=0.8, ls='--')
    ax.set_yticks(y, [label(p) for p in pops])
    ax.set_xlabel('Irregularity (CV of ISI)')
    ax.set_xlim(0, 1)
    save(fig, out_dir, 'irregularity_cv.png')


def power_spectrum(spikes, t0, t1, out_dir):
    """ Power spectrum of each layer's excitatory population rate, 1 ms bins (S5 Fig) """
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bins = np.arange(t0, t1 + 1, 1.0)
    shades = ['#E9A0AA', '#D1495B', '#9E2A3A', '#5E1622'] # One red per layer, light to dark (L2/3 -> L6)
    for pop, c in zip(['L23E', 'L4E', 'L5E', 'L6E'], shades):
        all_t = np.concatenate([t for t in spikes[pop].values()] or [np.array([])])
        rate = np.histogram(all_t, bins)[0] / SIZES[pop] * 1000 # Hz
        f, p = welch(rate - rate.mean(), fs=1000, nperseg=min(1024, len(rate)))
        ax.semilogy(f[1:], p[1:], color=c, lw=1.5, label=label(pop))
    ax.set_xlim(0, 150)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (Hz²/Hz)')
    ax.legend(frameon=False, ncol=2, fontsize=8)
    save(fig, out_dir, 'power_spectrum_E.png')


def classes(m):
    """ +1 for a change above +20%, -1 below -20%, 0 otherwise (the paper's red/blue/white) """
    return np.where(m > 20, 1, np.where(m < -20, -1, 0))


def matrix_plot(values, colors_of, cmap, title, out_dir, name, outline=None, legend=None):
    """ 16x16 matrix: rows = perturbed population, columns = observed population """
    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.imshow(colors_of, cmap=cmap, vmin=-1, vmax=1)
    n = len(MATRIX_POPS)
    for r in range(n):
        for c in range(n):
            v = values[r, c]
            diag = r == c
            if diag:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=INK))
            if outline is not None and outline[r, c] and not diag:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=INK, lw=1.5))
            ax.text(c, r, '' if np.isnan(v) else f'{v:.0f}', ha='center', va='center', fontsize=6,
                    color='white' if diag else INK)
    for k in range(4, n, 4): # Layer boundaries
        ax.axhline(k - 0.5, color=MUTED, lw=1)
        ax.axvline(k - 0.5, color=MUTED, lw=1)
    names = [label(p) for p in MATRIX_POPS]
    ax.set_xticks(range(n), names, rotation=90)
    ax.set_yticks(range(n), names)
    ax.set_xlabel('Observed population')
    ax.set_ylabel('Perturbed population (30 pA)')
    ax.set_title(title, color=INK, loc='left')
    ax.spines[:].set_visible(False)
    if legend:
        ax.legend(handles=legend, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    save(fig, out_dir, name)


def matrices(run_dir, out_dir):
    files = sorted(glob.glob(os.path.join(run_dir, 'metrics', 'perturbation_matrix_*.csv')))
    mats = {os.path.basename(f)[len('perturbation_matrix_'):-4]: pd.read_csv(f, index_col=0).loc[MATRIX_POPS, MATRIX_POPS].to_numpy()
            for f in files}
    up, down, none = '#F2A8A0', '#9FBFE3', 'white'
    for state, m in mats.items():
        matrix_plot(m, classes(m), ListedColormap([down, none, up]),
                    f'Perturbation matrix, {state} state: % rate change', out_dir, f'matrix_{state}.png',
                    legend=[Patch(color=up, label='> +20%'), Patch(color=down, label='< -20%'),
                            Patch(facecolor=none, edgecolor=MUTED, label='within ±20%')])
    if 'spont' in mats:
        more, less = '#F2A8A0', '#A8D5A2'
        for state, m in mats.items():
            if state == 'spont':
                continue
            diff_class = np.sign(classes(m) - classes(mats['spont']))
            flipped = np.sign(m) * np.sign(mats['spont']) < 0
            matrix_plot(m - mats['spont'], diff_class, ListedColormap([less, 'white', more]),
                        f'{state} vs spontaneous: difference in % change (box = sign flip)', out_dir,
                        f'comparison_{state}_vs_spont.png', outline=flipped,
                        legend=[Patch(color=more, label='Class up (e.g. none → >+20%)'),
                                Patch(color=less, label='Class down'),
                                Patch(facecolor='white', edgecolor=MUTED, label='Same class')])
            print(f'Frobenius norm |{state} - spont| = {np.linalg.norm(np.nan_to_num(m - mats["spont"])):.1f}')


def main():
    run_dir = sys.argv[1]
    warmup = float(sys.argv[2]) if len(sys.argv) > 2 else 500.
    out_dir = os.path.join(run_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)
    if os.path.isdir(os.path.join(run_dir, 'spike_times')) and os.listdir(os.path.join(run_dir, 'spike_times')):
        spikes = load_spikes(run_dir)
        t_end = cfg['t_sim']
        rates = neuron_rates(spikes, warmup, t_end)
        raster(spikes, cfg, out_dir)
        rate_bars(rates, out_dir)
        rate_boxplot(rates, out_dir)
        cv_bars(spikes, warmup, t_end, out_dir)
        power_spectrum(spikes, warmup, t_end, out_dir)
    matrices(run_dir, out_dir)


if __name__ == '__main__':
    main()
