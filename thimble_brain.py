# -*- coding: utf-8 -*-
# pragma: no cover

__author__ = "Veith Roethlingshoefer"

from hbp_nrp_cle.brainsim import simulator as sim
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_brain():

    sim.setup(timestep=0.1, min_delay=0.1, max_delay=20.0, threads=1, rng_seeds=[1234])

    SENSORPARAMS = {'cm': 0.025,
                    'v_rest': -60.5,
                    'tau_m': 10.,
                    'e_rev_E': 0.0,
                    'e_rev_I': -75.0,
                    'v_reset': -60.5,
                    'v_thresh': -60.0,
                    'tau_refrac': 10.0,
                    'tau_syn_E': 2.5,
                    'tau_syn_I': 2.5}

    SYNAPSE_PARAMS = {"weight": 0.5e-4,
                      "delay": 20.0,
                      'U': 1.0,
                      'tau_rec': 1.0,
                      'tau_facil': 1.0}

    cell_class = sim.IF_cond_alpha(**SENSORPARAMS)


    # Define the network structure: 12 neurons
    # 9 for the grid over the image, 3 for the index
    neurons = sim.Population(12, cellclass=cell_class)
    grid = sim.Population(size=9,
                          cellclass=cell_class)
    output = sim.Population(size=3,
                            cellclass=cell_class)

    synapse_type = sim.TsodyksMarkramSynapse(**SYNAPSE_PARAMS)

    # Connect neurons
    #Grid to output, per row
    """sim.Projection(presynaptic_population=neurons,
                   postsynaptic_population=neurons,
                   connector=sim.FromListConnector([(0, 9), (3, 9), (6, 9),
                                                    (1, 10), (4, 10), (7, 10),
                                                    (2, 11), (5, 11), (8, 11)]),
                   synapse_type=sim.StaticSynapse(weight=1.0),
                   receptor_type='excitatory')"""
    sim.Projection(presynaptic_population=neurons[0:3],
                   postsynaptic_population=neurons[9:12],
                   connector=sim.AllToAllConnector(),
                   synapse_type=sim.StaticSynapse(weight=1.0),
                   receptor_type='excitatory')

    sim.Projection(presynaptic_population=neurons[3:6],
               postsynaptic_population=neurons[9:12],
               connector=sim.AllToAllConnector(),
               synapse_type=sim.StaticSynapse(weight=1.0),
               receptor_type='excitatory')

    sim.Projection(presynaptic_population=neurons[6:9],
                  postsynaptic_population=neurons[9:12],
                  connector=sim.AllToAllConnector(),
                  synapse_type=sim.StaticSynapse(weight=1.0),
                  receptor_type='excitatory')

    # Inhibit between different outputs
    """sim.Projection(presynaptic_population=neurons,
                   postsynaptic_population=neurons,
                   connector=sim.FromListConnector([(9, 10), (9, 11),
                                                    (10, 9), (10, 11),
                                                    (11, 10), (11, 9)]),
                   synapse_type=sim.StaticSynapse(weight=1.0),
                   receptor_type='inhibitory')"""

    sim.initialize(neurons, v=neurons.get('v_rest'))
    return neurons

circuit = create_brain()
