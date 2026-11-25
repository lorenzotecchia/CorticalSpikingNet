from brian2 import *
import network_main
from network_parameters import net_dict, neuron_dict, receptors_dict, eqs_dict
from utils.utils import save_config, create_run_folder, setup_loggers
import hydra
from omegaconf import DictConfig
import os

set_device('cpp_standalone', build_on_run=False) # Brian2 generates the simulation as C++; Network_main.build() compiles it once
prefs.devices.cpp_standalone.extra_make_args_unix = ['-j8'] # Default '-j' starts one compiler per file (~1200 here) at once and exhausts RAM
defaultclock.dt = 0.1*ms # Time resolution of the simulation
seed(net_dict['Seed']) # Brian2's seed also seeds the generated C++ code; np.random.seed alone does not
@hydra.main(version_base=None, config_path='config', config_name='config.yaml')
def main(cfg: DictConfig):
    t_sim = cfg.t_sim * ms # Simulation time - needs to be in ms in config file
    EXPERIMENT_PATH = os.environ.get('EXPERIMENT_PATH', 'runs') # Path of directory where experiments are saved
    run_dir, raster_plots_dir, spike_times_dir, metrics_dir = create_run_folder(EXPERIMENT_PATH, cfg.experiment_type)
    save_config(cfg, run_dir)
    log = setup_loggers(run_dir)

    net = network_main.Network_main(net_dict, neuron_dict, receptors_dict, eqs_dict, log)
    net.create()
    net.connect(cfg)
    net.build(t_sim)
    net.run_with_input('results', cfg.input.state, cfg.input.pert, cfg.input.t_state*ms, cfg.input.t_pert*ms)
    net.plot_spikes(raster_plots_dir)
    net.firing_rates(t_sim, metrics_dir)
    net.write_data(spike_times_dir)

if __name__ == "__main__":
    main()
