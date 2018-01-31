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

    cell_class = sim.IF_cond_alpha(**SENSORPARAMS)

    neurons = sim.Population(17, cellclass=cell_class)

    xyt = neurons[0:6]
    sxt = neurons[6:12]
    normed = neurons[12:15]
    minxyt_1 = neurons[15:17]


    sim.Projection(presynaptic_population=xyt,
                   postsynaptic_population=sxt,
                   connector=sim.OneToOneConnector(),
                   synapse_type=sim.StaticSynapse(weight=1.0),
                   receptor_type='excitatory')

    sim.Projection(presynaptic_population=mixyt_1,
                   postsynaptic_population=sxt,
                   connector=sim.FromListConnector([(0, 0), (0, 2), (0, 4),
                                                    (1, 1), (1, 3), (1, 5)]),
                   synapse_type=sim.StaticSynapse(weight=1.0),
                   receptor_type='inhibitory')

    sim.Projection(presynaptic_population=sxt,
               postsynaptic_population=normed,
               connector=sim.FromListConnector([(0, 0), (1, 0),
                                                (2, 1), (3, 1),
                                                (4, 2), (5, 2)]),
               synapse_type=sim.StaticSynapse(weight=1.0),
               receptor_type='excitatory')

    sim.initialize(neurons, v=neurons.get('v_rest'))
    return neurons

circuit = create_brain()
xyt = circuit[0:6]
sxt = circuit[6:12]
normed = circuit[12:15]
minxyt_1 = circuit[15:17]
