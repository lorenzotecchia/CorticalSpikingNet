from brian2 import *
import matplotlib
matplotlib.use('Agg') # Only saves PNGs; also keeps macOS GUI libraries out of the process, which makes fork() safe
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import pandas as pd
from network_parameters import input_states


class Network_main:
    """"
    Parameters
    ---------
    net_dict : dictionary
         Contains parameters specific to overall network (see: 'network_parameters.py').
    neuron_dict : dictionary
         Contains parameters specific to neuron used (see: 'network_parameters.py').
    receptors_dict : dictionary
         Contains parameters specific to AMPA, GABA, NMDA receptors (see: 'network_parameters.py').
    eqs_dict : dictionary
         Contains equations of neurons and specific to AMPA, GABA, NMDA receptors (see: 'network_parameters.py').
    """
    def __init__(self, net_dict, neuron_dict, receptors_dict, eqs_dict, log):
        self.net_dict = net_dict
        self.neuron_dict = neuron_dict
        self.receptors_dict = receptors_dict
        self.eqs_dict = eqs_dict
        self.log = log
        
        self.log.info('---------------------------------')
        self.log.info('      CORTICAL COLUMN MODEL')
        self.log.info('---------------------------------')
        self.num_pops = len(self.net_dict['populations']) # Number of populations
        self.log.info('Number of populations: %s', self.num_pops)
        self.num_neurons = self.net_dict['num_neurons'] # Number of neurons
        self.log.info(self.net_dict['num_neurons'])

    def create(self):
        """ Creates network populations and input.

        Neuronal populations and devices (recording and generators) are created.

        """
        self.__create_neuronal_populations()
        self.__create_poisson_bg_input()

    def connect(self, cfg):
        """ Connects the population and devices (recording and generators).

        Neuronal populations are connected between and within themselves,
        as well as generator devices and recording devices

        """
        
        self.__connect_poisson_bg_input()
        self.__connect_neuronal_populations(cfg)
        self.__connect_recording_devices()

    def build(self, t_sim):
        """ Compiles the network into a C++ program that simulates t_sim.

        Compiling happens once; every call of run_with_input() reuses the program.
        """
        self.log.info('-----------------------------')
        self.log.info('Building')
        self.log.info('-----------------------------')
        self.net_run = Network(self.spike_mon, self.pops, self.Poisson_groups, self.S_Poisson,
                                self.S_ampa, self.S_gaba, self.S_nmda) # Every object must be listed here to be simulated
        self.net_run.run(t_sim) # With build_on_run=False this only records the run; nothing is simulated yet
        device.build(directory='standalone_build', run=False)
        self.log.info('Finished building')

    def run_with_input(self, results_dir, state='spont', pert_pop=None, t_state=0*ms, t_pert=0*ms):
        """ Runs the compiled program once with the given external input.

        Parameters
        ----------
        results_dir: Folder for this run's raw results, relative to standalone_build (one per parallel run).
        state: Network state, a key of input_states in network_parameters.py (spont, FF, FB, FF_FB).
        pert_pop: Name of the population that gets the perturbation (pert_amp), or None.
        t_state, t_pert: Times at which the state input and the perturbation switch on.
        """
        unknown = set(input_states[state]) | ({pert_pop} - {None})
        unknown -= set(self.net_dict['populations'])
        assert not unknown, f'Unknown populations: {unknown}'
        run_args = {}
        for name, pop in zip(self.net_dict['populations'], self.pops):
            run_args[pop.t_state] = t_state
            run_args[pop.t_pert] = t_pert
            if name in input_states[state]:
                amp, fraction = input_states[state][name]
                I_state = np.zeros(len(pop))*pA
                I_state[:int(len(pop)*fraction)] = amp # First cells of the population, as in the original code
                run_args[pop.I_state] = I_state
            if name == pert_pop:
                run_args[pop.I_pert] = self.net_dict['pert_amp']
        self.log.info('Running: state %s, perturbation %s', state, pert_pop)
        device.run(results_directory=results_dir, run_args=run_args, with_output=False)

    def window_rates(self, t_start, t_end):
        """ Mean firing rate (Hz) of each population between t_start and t_end, as an array. """
        return np.array([np.sum((mon.t >= t_start) & (mon.t < t_end)) / (len(pop)*float(t_end - t_start))
                         for mon, pop in zip(self.spike_mon, self.pops)])


    def __create_neuronal_populations(self):
        """ Creates the neuronal populations with parameters defined in 'network_parameters.py'.

            Stores them in a list, pops
        """
        self.log.info('Creating neuronal populations')
        self.pops = []
        for m in np.arange(self.num_pops):
            Vth = self.neuron_dict['V_th'][m]
            Vrest = self.neuron_dict['V_reset'][m]
            V_L_i = self.neuron_dict['V_L'][m]
            C_m_i = self.neuron_dict['C_m'][m]
            g_L_i = self.neuron_dict['g_L'][m]
            population = NeuronGroup(self.net_dict['num_neurons'][m], model=self.eqs_dict['eqs_neuron'], threshold='v > v_th', # repr method to convert a Quantity object, which is a physical quantity with units, to a string that can be used in a string format expression.
                                    reset='v = v_rest', refractory=self.neuron_dict['tau_ref'][m], method='euler')
            population.v_th = Vth
            population.v_rest = Vrest
            population.v = self.neuron_dict['V_0'][m]
            population.V_L = V_L_i # Set V_l for each population as in network_parameters
            population.C_m = C_m_i # Set C_m for each population as in network_parameters
            population.g_L = g_L_i # Set g_L for each population as in network_parameters
            population.V_I = self.receptors_dict['V_I'][m] # V_I for GABA
            self.pops.append(population)

    def __create_poisson_bg_input(self):
        """ Creates Poisson background input ith parameters 
        as specified in 'network_parameters.py'.

            The same number of generators is created as neuronal populations.
            The number of neurons in each Poisson generator is equal to the
            number of neurons in the population it targets.

        """
        self.log.info('-----------------------------')
        self.log.info('Creating Poisson background input')
        self.log.info('-----------------------------')

        self.Poisson_groups = [] # List to hold Poisson background generators
        for n in np.arange(self.num_pops):
            Poisson_groups = PoissonGroup(self.net_dict['num_neurons'][n], rates=self.net_dict['bg_rate'][n])
            self.Poisson_groups.append(Poisson_groups)
    
    def __connect_neuronal_populations(self, cfg):
        """ Connects neuronal populations recurrently, with parameters
        as specified in 'network_parameters.py'.

            There are E-E, E-I, I-E and I-I connections.

        """
        self.log.info('-----------------------------')
        self.log.info('Connecting synapses')
        self.log.info('-----------------------------')

        self.S_ampa = []
        self.S_nmda = []
        self.S_gaba = []
        # Excitatory sources (index -> NMDA equation of their layer); all other populations are inhibitory (GABA)
        nmda_eqs = {1: 'eqs_s_nmda_l23', 5: 'eqs_s_nmda_l4', 9: 'eqs_s_nmda_l5', 13: 'eqs_s_nmda_l6'}
        for n, target_pop in enumerate(self.pops):
            for m, source_pop in enumerate(self.pops):
                p = self.net_dict['connect_probs'][n][m] * cfg.synaptic_density # Local copy: net_dict stays unchanged between runs
                strength = self.net_dict['synaptic_strength'][n][m]
                if p == 0 or strength == 0:
                    continue # No connection: skip it instead of building an empty Synapses object
                if m in nmda_eqs: # Excitatory: AMPA and NMDA synapses, drawn with 0.8*p and 0.2*p
                    receptors = [('w : 1', 's_ampa_post += w', self.net_dict['proportion_AMPA'], self.S_ampa),
                                 (self.eqs_dict[nmda_eqs[m]], 'x += 1', self.net_dict['proportion_NMDA'], self.S_nmda)]
                else: # Inhibitory: GABA synapses
                    receptors = [('w : 1', 's_gaba_post += w', 1., self.S_gaba)]
                for model, on_pre, fraction, synapse_list in receptors:
                    S = Synapses(source_pop, target_pop, model=model, on_pre=on_pre, method='euler')
                    S.connect(condition='i != j' if m == n else None, p=fraction*p) # i != j prevents auto-synapses
                    S.w = self.net_dict['global_g']*strength/(self.net_dict['num_neurons'][m]*fraction*p) # Eq. 11 of Moreni et al. (2025)
                    S.delay = self.net_dict['delay']
                    synapse_list.append(S)
        self.log.info('Synapse objects: %s AMPA, %s NMDA, %s GABA', len(self.S_ampa), len(self.S_nmda), len(self.S_gaba))


    def __connect_poisson_bg_input(self):
        """ Connects generator devices as specified in 'network_parameters.py'.

            Connects Poisson background generator to each population.
        
        """
        self.log.info('Connecting Poisson background input')
        self.log.info('-----------------------------')

        self.S_Poisson = [] # List for Poisson background synapses
        for n in np.arange(self.num_pops):
            self.log.info('Connecting Poisson %s', n)
            S_Poisson = Synapses(self.Poisson_groups[n], self.pops[n], on_pre='s_ampa_ext += 1', method='euler')
            self.S_Poisson.append(S_Poisson)
            self.S_Poisson[n].connect(j='i') # Connects Poisson one-to-one

    def __connect_recording_devices(self):
        """ Creates the spike/voltage recording devices for each population.

            It automatically connects to the population specified.

        """
        self.log.info('-----------------------------')
        self.log.info('Connecting recording devices')

        self.spike_mon = [SpikeMonitor(pop) for pop in self.pops] # Record from all neurons in each population
    
    def plot_spikes(self, raster_plots_dir):
        """ Plots raster graphs after simulation is run.

        """
        list_ex = [1, 5, 9 , 13] # Indexes corresponding to excitatory populations
        color = ''
        for i in range(len(self.pops)):
            pop = self.net_dict['populations'][i]
            if i in list_ex:
                color = '.r' # Red for excitatory
            else:
                color = '.b' # Blue for inhibitory
            plt.figure(i)
            plt.plot(self.spike_mon[i].t/ms, self.spike_mon[i].i, color, label=pop)
            xlabel('Time (ms)')
            ylabel('Neuron index')
            legend()
            plt.savefig(os.path.join(raster_plots_dir, f"5000_spikes_population_{i}.png"))
            plt.close()

    def firing_rates(self, t_sim, metrics_dir):
        """ Calculates and prints mean firing rates in spikes/s for each population.

        """
        avg_firing_rates = pd.DataFrame(columns=['Population', 'Avg Firing Rate Hz'])
        avg_firing_rates_path = os.path.join(metrics_dir, 'avg_firing_rates.csv')
        for i in range(self.num_pops):
            mean_rate = self.spike_mon[i].num_spikes/(self.net_dict['num_neurons'][i]*t_sim)
            self.log.info(f"Mean rate of population {self.net_dict['populations'][i]}: {mean_rate:.2f} Hz")

            new_row = [self.net_dict['populations'][i], f'{mean_rate:.2f}']
            avg_firing_rates.loc[len(avg_firing_rates)] = new_row
        avg_firing_rates.to_csv(avg_firing_rates_path, index=False)

    def write_data(self, spike_times_dir):
        """ Creates files with spike data.
            For each population, create a csv file which has one column
            with the index of the neuron and the second column with
            a list containing the spike times (in ms) of that neuron.
            Saves them in a folder called Spike_times of the current run.
        """
        for n in range(self.num_pops):
            with open(f"{spike_times_dir}/{self.net_dict['populations'][n]}.csv", 'w') as csv_file:
                writer = csv.writer(csv_file)
                for neuron, times in self.spike_mon[n].spike_trains().items():
                    writer.writerow([neuron, np.round(times/ms, 1).tolist()]) # Spike times in ms, as plain floats
