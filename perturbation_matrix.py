"""Perturbation matrices of Moreni et al. (2025): in each network state, give pert_amp to one population at a time
and record how the rate of every population changes (window after the perturbation vs the same window before it).

The network is compiled once; each simulation reruns the compiled program with different run_args,
several at the same time. Settings: 'sweep' in config/config.yaml. Output: metrics/perturbation_matrix_<state>.csv
(rows: perturbed population, columns: observed population, values: % rate change) and baseline_rates.csv.
"""
import multiprocessing
import os

import hydra
import numpy as np
import pandas as pd
from brian2 import ms
from omegaconf import DictConfig

import simulate  # Sets up the C++ standalone device, time step and seed
import network_main
from network_parameters import net_dict, neuron_dict, receptors_dict, eqs_dict
from utils.utils import save_config, create_run_folder, setup_loggers

NET = None # Built network; forked workers inherit it
CFG = None


def run_one(job):
    """ One simulation: rates of all populations before and after the perturbation switches on. """
    state, pert_pop = job
    warmup, window = CFG.sweep.warmup*ms, CFG.sweep.window*ms
    t_pert = warmup + window
    NET.run_with_input(os.path.join('results', f'{state}_{pert_pop}'), state, pert_pop, 0*ms, t_pert)
    return state, pert_pop, NET.window_rates(warmup, t_pert), NET.window_rates(t_pert, t_pert + window)


@hydra.main(version_base=None, config_path='config', config_name='config.yaml')
def main(cfg: DictConfig):
    global NET, CFG
    CFG = cfg
    run_dir, _, _, metrics_dir = create_run_folder(os.environ.get('EXPERIMENT_PATH', 'runs'), cfg.experiment_type)
    save_config(cfg, run_dir)
    log = setup_loggers(run_dir)

    NET = network_main.Network_main(net_dict, neuron_dict, receptors_dict, eqs_dict, log)
    NET.create()
    NET.connect(cfg)
    NET.build((cfg.sweep.warmup + 2*cfg.sweep.window)*ms)

    pops = net_dict['populations']
    jobs = [(state, pert_pop) for state in cfg.sweep.states for pert_pop in pops]
    changes = {state: {} for state in cfg.sweep.states}
    baselines = {}
    # fork: workers inherit the built network; one fresh worker per job, so no state carries over between runs
    with multiprocessing.get_context('fork').Pool(cfg.sweep.workers, maxtasksperchild=1) as pool:
        for state, pert_pop, before, after in pool.imap_unordered(run_one, jobs):
            changes[state][pert_pop] = 100*(after - before)/np.where(before > 0, before, np.nan) # NaN if silent before
            baselines[state] = before # Same seed in every run, so the baseline of a state is the same in all its runs
            log.info('Done: state %s, perturbation %s', state, pert_pop)

    for state, rows in changes.items():
        pd.DataFrame([rows[p] for p in pops], index=pops, columns=pops).round(2).to_csv(
            os.path.join(metrics_dir, f'perturbation_matrix_{state}.csv'))
    pd.DataFrame(baselines, index=pops).round(3).to_csv(os.path.join(metrics_dir, 'baseline_rates.csv'))
    log.info('Saved perturbation matrices in %s', metrics_dir)


if __name__ == "__main__":
    main()
