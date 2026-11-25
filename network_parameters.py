# File with the Neuronal and network parameters
from brian2 import *
import numpy as np


N_tot = 5000 # Total number of neurons
N_1 = 0.0192574218*N_tot # Number of neurons in layer 1
N_23 = 0.291088453*N_tot # Number of neurons in layer 23
N_4 = 0.237625904*N_tot # Number of neurons in layer 4
N_5 = 0.17425693*N_tot # Number of neurons in layer 5
N_6 = N_tot - N_23 - N_4 - N_5 # Number of neurons in layer 6
proportion_exc = 0.85 # Proportion of excitatory neurons in each layer
proportion_inh = 0.15 # Proportion of inhibitory neurons in each layer
L_1_vip = int(N_1) # Number of VIP neurons in layer 1
L_23_exc = int(proportion_exc*N_23) # Number of excitatory neurons in layer 23
L_23_pv = int(0.295918*proportion_inh*N_23) # Number of PV neurons in layer 23
L_23_sst = int(0.214286*proportion_inh*N_23) # Number of SST neurons in layer 23
L_23_vip = int(0.489796*proportion_inh*N_23) # Number of VIP neurons in layer 23
L_4_exc = int(proportion_exc*N_4) # Number of excitatory neurons in layer 4
L_4_pv = int(0.552381*proportion_inh*N_4) # Number of PV neurons in layer 4
L_4_sst = int(0.295238*proportion_inh*N_4) # Number of SST neurons in layer 4
L_4_vip = int(0.152381*proportion_inh*N_4) # Number of VIP neurons in layer 4
L_5_exc = int(proportion_exc*N_5) # Number of excitatory neurons in layer 5
L_5_pv = int(0.485714*proportion_inh*N_5) # Number of PV neurons in layer 5
L_5_sst = int(0.428571*proportion_inh*N_5) # Number of SST neurons in layer 5
L_5_vip = int(0.085714*proportion_inh*N_5) # Number of VIP neurons in layer 5
L_6_exc = int(proportion_exc*N_6) # Number of excitatory neurons in layer 6
L_6_pv = int(0.458333*proportion_inh*N_6) # Number of PV neurons in layer 6
L_6_sst = int(0.458333*proportion_inh*N_6) # Number of SST neurons in layer 6
L_6_vip = int(0.083333*proportion_inh*N_6) # Number of VIP neurons in layer 6

net_dict = {
    # Simulation seed
    'Seed': 55,
    # Names of the populations simulated
    'populations': ['L1VIP', 'L23E', 'L23PV', 'L23SST', 'L23VIP', 'L4E', 'L4PV', 'L4SST', 'L4VIP', 'L5E', 'L5PV', 'L5SST', 'L5VIP', 'L6E', 'L6PV', 'L6SST', 'L6VIP'],
    # Number of neurons in the populations simulated (same order as above)
    'num_neurons':
        np.array([L_1_vip, L_23_exc, L_23_pv, L_23_sst, L_23_vip, L_4_exc, L_4_pv, L_4_sst, L_4_vip, L_5_exc, L_5_pv, L_5_sst, L_5_vip, L_6_exc, L_6_pv, L_6_sst, L_6_vip]),
    # Connection probability matrix
    'connect_probs':
        np.array([
                [0.656, 0., 0.024, 0.279, 0., 0., 0., 0.241, 0., 0.017, 0., 0.203, 0., 0., 0., 0., 0.], # L1VIP - target
                [0.356, 0.16, 0.411, 0.424, 0.087, 0.14, 0.25, 0.25, 0.25, 0.021, 0., 0.169, 0., 0., 0.1, 0., 0.], # L23E
                [0.093, 0.395, 0.451, 0.857, 0.02, 0.1, 0.05, 0.05, 0.05, 0.05, 0.102, 0., 0., 0., 0., 0., 0.], # L23PV
                [0.068, 0.182, 0.03, 0.082, 0.625, 0.1, 0.05, 0.05, 0.05, 0.05, 0., 0.017, 0., 0., 0., 0., 0.], # L23SST
                [0.464, 0.105, 0.22, 0.77, 0.028, 0.1, 0.05, 0.05, 0.05, 0.05, 0., 0., 0., 0., 0., 0., 0.], # L23VIP
                [0.148, 0.016, 0.05, 0.05, 0.05, 0.243, 0.437, 0.351, 0.351, 0.007, 0., 0.056, 0.03, 0., 0.1, 0., 0.], # L4E
                [0., 0.083, 0.05, 0.05, 0.05, 0.43, 0.451, 0.857, 0.02, 0.05, 0.034, 0.03, 0.03, 0., 0., 0., 0.], # L4PV
                [0., 0.083, 0.05, 0.05, 0.05, 0.571, 0.03, 0.082, 0.625, 0.05, 0.03, 0.006, 0.03, 0., 0., 0., 0.], # L4SST
                [0, 0.083, 0.05, 0.05, 0.05, 0.571, 0.22, 0.77, 0.028, 0.05, 0.03, 0.03, 0.03, 0., 0., 0., 0.], # L4VIP
                [0.148, 0.083, 0.07, 0.021, 0., 0.104, 0.088, 0.026, 0., 0.116, 0.455, 0.317, 0.125, 0.012, 0.1, 0.03, 0.03], # L5E
                [0., 0.081, 0.073, 0., 0., 0.101, 0.091, 0.03, 0.03, 0.083, 0.361, 0.857, 0.02, 0.01, 0.03, 0.03, 0.03], # L5PV
                [0., 0.102, 0., 0., 0., 0.128, 0.03, 0., 0.03, 0.063, 0.03, 0.04, 0.625, 0.01, 0.03, 0.03, 0.03], # L5SST
                [0., 0., 0., 0., 0., 0.05, 0.03, 0.03, 0.03, 0.105, 0.22, 0.77, 0.02, 0.01, 0.03, 0.03, 0.03], # L5VIP
                [0.148, 0., 0., 0., 0., 0.032, 0., 0., 0., 0.047, 0.03, 0.03, 0.03, 0.026, 0.1, 0.1, 0.1], # L6E
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.03, 0.01, 0.01, 0.01, 0.145, 0.08, 0.05, 0.05], # L6PV
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.03, 0.01, 0.01, 0.01, 0.1, 0.1, 0.05, 0.05], # L6SST
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.03, 0.01, 0.01, 0.01, 0.1, 0.08, 0.05, 0.03]]), # L6VIP
    # Synaptic strength matrix
    'synaptic_strength':
        np.array([
                [1.73, 0., 0.37, 0.47, 0., 0., 0., 0.39, 0., 0.76, 0., 0.31, 0., 0., 0., 0., 0.], # L1VIP
                [0.53, 0.36, 0.48, 0.31, 0.28, 0.78, 0.56, 0.3, 0.29, 0.47, 0., 0.25, 0., 0., 0.81, 0., 0.], # L23E
                [0.48, 1.49, 0.68, 0.5, 0.18, 1.39, 0.68, 0.5, 0.18, 1.25, 0.51, 0., 0., 0., 0., 0., 0.], # L23PV
                [0.57, 0.86, 0.42, 0.15, 0.32, 0.69, 0.42, 0.15, 0.32, 0.52, 0., 0.39, 0., 0., 0., 0., 0.], # L23SST
                [0.78, 1.31, 0.41, 0.52, 0.37, 0.91, 0.41, 0.52, 0.37, 0.91, 0., 0., 0., 0., 0., 0., 0.], # L23VIP
                [0.42, 0.34, 0.56, 0.3, 0.29, 0.83, 0.64, 0.29, 0.29, 0.38, 0., 0.28, 0.29, 0., 0.81, 0., 0.], # L4E
                [0., 1.39, 0.68, 0.5, 0.18, 1.29, 0.68, 0.5, 0.18, 1.25, 0.94, 0.45, 0.18, 0., 0., 0., 0.], # L4PV
                [0., 0.69, 0.42, 0.15, 0.32, 0.51, 0.42, 0.15, 0.32, 0.52, 0.42, 0.28, 0.33, 0., 0., 0., 0.], # L4SST
                [0., 0.91, 0.41, 0.52, 0.37, 0.51, 0.41, 0.52, 0.37, 0.91, 0.41, 0.52, 0.37, 0., 0., 0., 0.], # L4VIP
                [0.42, 0.74, 0.2, 0.22, 0., 0.63, 0.73, 0.28, 0., 0.75, 0.81, 0.27, 0.28, 0.23, 0.81, 0.27, 0.28], # L5E
                [0., 1.32, 0.79, 0., 0., 1.25, 0.94, 0.45, 0.18, 1.2, 1.19, 0.4, 0.18, 2.5, 1.19, 0.4, 0.18], # L5PV
                [0., 0.53, 0., 0., 0., 0.52, 0.42, 0.28, 0.33, 0.52, 0.41, 0.4, 0.33, 0.52, 0.41, 0.4, 0.33], # L5SST
                [0., 0., 0., 0., 0., 0.91, 0.41, 0.52, 0.37, 1.31, 0.41, 0.52, 0.37, 1.31, 0.41, 0.52, 0.37], # L5VIP
                [0.42, 0., 0., 0., 0., 0.96, 0., 0., 0., 0.4, 0.81, 0.27, 0.28, 0.94, 0.81, 0.27, 0.28], # L6E
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 2.5, 1.19, 0.4, 0.18, 3.8, 1.19, 0.4, 0.18], # L6PV
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.52, 0.41, 0.4, 0.33, 0.52, 0.41, 0.4, 0.33], # L6SST
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.31, 0.41, 0.52, 0.37, 1.31, 0.41, 0.52, 0.37]]), # L6VIP    
    # Factor to multiply the weight of the connections globally
    'global_g' : 5.,
    # Probablity of AMPA projections from excitatory is 0.8 of total probability of connection
    'proportion_AMPA': 0.8,
    # Probablity of NMDA projections from excitatory is 0.2 of total probability of connection
    'proportion_NMDA': 0.2,
    # There is only one synaptic delay in Giulia's column model, 2*ms
    'delay' : 2.*ms,
    # Rate of Poisson generator for background noise, same order as in 'populations'
    'bg_rate':
        np.array([650., 930., 1460., 870., 1405., 890., 1980., 2105., 240., 4740., 930., 530., 870., 1770., 1170., 885., 1620.])*Hz,
    # Amplitude of the perturbation given to one population at a time (perturbation matrices)
    'pert_amp': 30*pA,
}

# Network states of Moreni et al. (2025): population -> (input amplitude, fraction of its cells that receive it).
# The input goes to the first cells of the population, as in the original code.
input_states = {
    'spont': {},
    'FF': {'L4E': (150*pA, 0.25), 'L4PV': (150*pA, 0.05)},
    'FB': {'L5E': (150*pA, 0.25), 'L5PV': (150*pA, 0.05)},
    'FF_FB': {'L4E': (150*pA, 0.25), 'L4PV': (150*pA, 0.05), 'L5E': (150*pA, 0.25), 'L5PV': (150*pA, 0.05)},
    # The four feedback states of Fig 5
    'FB23': {'L23E': (150*pA, 0.25), 'L23PV': (150*pA, 0.05)},
    'FB23_5': {'L23E': (150*pA, 0.25), 'L23PV': (150*pA, 0.05), 'L5E': (150*pA, 0.25), 'L5PV': (150*pA, 0.05)},
    'FB6': {'L6E': (150*pA, 0.25), 'L6PV': (150*pA, 0.05)},
    'FB23_5_6': {'L23E': (150*pA, 0.25), 'L23PV': (150*pA, 0.05), 'L5E': (150*pA, 0.25), 'L5PV': (150*pA, 0.05),
                 'L6E': (150*pA, 0.25), 'L6PV': (150*pA, 0.05)},
}

neuron_dict = {
    # Initial membrane potential of neuronal populations
    'V_0': [-65.5, -80.97, -82.35, -69.16, -67.94, -72.53, -70.45, -74.2, -63.14, -68.28, -77.5, -70.01, -72., -77.5, -76.42, -62.99, -78.85]*mV,
    # Threshold to spike
    'V_th': [-40.2, -40.53, -56.32, -39.95, -41.34, -47.63, -44.23, -44.07, -40.89, -40.55, -51.2, -47.38, -51.2, -42.31, -49.06, -37.19, -44.81]*mV,
    # Reset voltage after a spike
    'V_reset': [-65.5, -80.97, -82.35, -69.16, -67.94, -72.53, -70.45, -74.2, -63.14, -68.28, -77.5, -70.01, -72., -77.5, -76.42, -62.99, -78.85]*mV,
    # Resting membrane potential
    'V_L': [-65.5, -80.97, -82.35, -69.16, -67.94, -72.53, -70.45, -74.2, -63.14, -68.28, -77.5, -70.01, -72., -77.5, -76.42, -62.99, -78.85]*mV,
    # Membrane capacitance
    'C_m': [37.11, 123.41, 70.95, 82.34, 41.23, 80.16, 81.21, 132.86, 40.3, 149.43, 70.9, 52.32, 59.29, 99.96, 49.65, 96.09, 65.87]*pF,
    # Membrane conductance
    'g_L': [4.07, 2.47, 9.49, 3.17, 6.4, 5.16, 9.19, 7.96, 1.87, 16.66, 5.21, 3.43, 6.52, 5.88, 6.86, 2.99, 6.09]*nS,
    # Refractory time
    'tau_ref': [3.5, 3., 1.26, 1.85, 2.75, 4.4, 1.5, 2.2, 2.4, 4.25, 1.85, 1.9, 2.55, 3.3, 1.65, 2.1, 2.85]*ms
}

receptors_dict = {
    # Equilibrium potential for AMPA and NMDA
    'V_E': 0., # mV

    ### AMPA ###

    # Time constant AMPA
    'tau_ampa': 2., # ms
    # Conductance of AMPA for external inputs
    'g_ampa_ext': 1., # nS
    # Conductance of AMPA for population connections
    'g_ampa_rec': 1., #nS

    ### NMDA ###

    # Rise time constant NMDA
    'tau_nmda_rise': 2., # ms
    # Decay time constant NMDA
    'tau_nmda_decay': 80., # ms
    # Conductance of NMDA receptors
    'g_nmda': 1., # nS
    # Magnesium concentration for NMDA receptors
    'Mg2': 1.,
    # Alpha for NMDA receptors
    'alpha': 0.5, # kHz

    ### GABA ###

    # Equilibrium potential for GABA receptors
    'V_I': [-65.5, -80.97, -82.35, -69.16, -67.94, -72.53, -70.45, -74.2, -63.14, -68.28, -77.5, -70.01, -72., -77.5, -76.42, -62.99, -78.85]*mV,
    # Time constant GABA
    'tau_gaba': 5., # ms
    # Conductance of GABA receptors
    'g_gaba': 1., # nS
}

eqs_dict = {
    # Equation for neuron
    'eqs_neuron': Equations(f'''

        dv/dt = (-g_L*(v - V_L) - I_syn + I_ext)/C_m: volt (unless refractory)
        I_syn = I_ampa_ext + I_ampa_rec + I_nmda + I_gaba : amp

        # External input (Iext in the paper), POSITIVE = excitatory. Two inputs that switch on at their own time:
        # a network state (FF/FB, see input_states) and a perturbation. All set per run via run_args, no recompiling.
        I_ext = I_state*int(t >= t_state) + I_pert*int(t >= t_pert) : amp
        I_state : amp
        I_pert : amp
        t_state : second (shared)
        t_pert : second (shared)

        V_L : volt
        g_L : siemens
        C_m : farad
        v_th : volt
        v_rest : volt

        I_ampa_ext = {receptors_dict['g_ampa_ext']}*nS*(v - {receptors_dict['V_E']}*mV)*s_ampa_ext : amp
        ds_ampa_ext/dt = -s_ampa_ext/({receptors_dict['tau_ampa']}*ms) : 1

        # Recurrent AMPA and GABA live on the neuron: one decaying variable per receptor that jumps by w at each
        # incoming spike. Exactly equal to the sum of per-synapse variables (Eqs. 4, 5, 7, 10), since all share one tau.
        I_ampa_rec = {receptors_dict['g_ampa_rec']}*nS*(v - {receptors_dict['V_E']}*mV)*s_ampa : amp
        ds_ampa/dt = -s_ampa/({receptors_dict['tau_ampa']}*ms) : 1

        I_gaba = {receptors_dict['g_gaba']}*nS*(v - V_I)*s_gaba : amp
        V_I : volt
        ds_gaba/dt = -s_gaba/({receptors_dict['tau_gaba']}*ms) : 1

        I_nmda = ({receptors_dict['g_nmda']}*nS*(v - {receptors_dict['V_E']}*mV)/(1 + {receptors_dict['Mg2']}*exp(-0.062*v/mV)/3.57))*s_nmda_tot : amp
        s_nmda_tot = s_nmda_tot_l23 + s_nmda_tot_l4 + s_nmda_tot_l5 + s_nmda_tot_l6 : 1
        s_nmda_tot_l23 : 1
        s_nmda_tot_l4 : 1
        s_nmda_tot_l5 : 1
        s_nmda_tot_l6 : 1

        '''),

### NMDA ###

    # Equation for NMDA L23
    'eqs_s_nmda_l23': Equations(f'''

        s_nmda_tot_l23_post = w*s_nmda : 1 (summed)
        ds_nmda/dt = -s_nmda/({receptors_dict['tau_nmda_decay']}*ms) + {receptors_dict['alpha']}*kHz*x*(1 - s_nmda) : 1 (clock-driven)
        dx/dt = -x/({receptors_dict['tau_nmda_rise']}*ms) : 1 (clock-driven)
        w : 1

        '''),

    # Equation for NMDA L4
    'eqs_s_nmda_l4': Equations(f'''

        s_nmda_tot_l4_post = w*s_nmda : 1 (summed)
        ds_nmda/dt = -s_nmda/({receptors_dict['tau_nmda_decay']}*ms) + {receptors_dict['alpha']}*kHz*x*(1 - s_nmda) : 1 (clock-driven)
        dx/dt = -x/({receptors_dict['tau_nmda_rise']}*ms) : 1 (clock-driven)
        w : 1

        '''),


    # Equation for NMDA L5
    'eqs_s_nmda_l5': Equations(f'''

        s_nmda_tot_l5_post = w*s_nmda : 1 (summed)
        ds_nmda/dt = -s_nmda/({receptors_dict['tau_nmda_decay']}*ms) + {receptors_dict['alpha']}*kHz*x*(1 - s_nmda) : 1 (clock-driven)
        dx/dt = -x/({receptors_dict['tau_nmda_rise']}*ms) : 1 (clock-driven)
        w : 1

        '''),

    # Equation for NMDA L6
    'eqs_s_nmda_l6': Equations(f'''

        s_nmda_tot_l6_post = w*s_nmda : 1 (summed)
        ds_nmda/dt = -s_nmda/({receptors_dict['tau_nmda_decay']}*ms) + {receptors_dict['alpha']}*kHz*x*(1 - s_nmda) : 1 (clock-driven)
        dx/dt = -x/({receptors_dict['tau_nmda_rise']}*ms) : 1 (clock-driven)
        w : 1

        ''')
}