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
                    'tau_refrac': 0.1,
                    'tau_syn_E': 2.5,
                    'tau_syn_I': 2.5}

    depressing_syn = sim.TsodyksMarkramSynapse(U=0.5, tau_rec=800.0, tau_facil=0.0,
                                               weight=0.01, delay=0.1)
    facilitating_syn = sim.TsodyksMarkramSynapse(U=0.5, tau_rec=800.0,
                                                 tau_facil=1000.0, weight=0.01,
                                                 delay=0.1)

    cell_class = sim.IF_cond_exp(**SENSORPARAMS)

    retina_neurons = 30, 40
    total_retina = int(np.prod(retina_neurons))
    neurons = sim.Population(11 + total_retina, cellclass=cell_class)
    coords = np.empty(retina_neurons + (3,), dtype=np.float32)
    rows = np.arange(retina_neurons[0])
    cols = np.arange(retina_neurons[1])
    coords[..., 0 ] = rows[:, None]
    coords[..., 1] = cols
    coords[..., 2] = 0
    positions = np.transpose(coords).reshape((3, -1))
    positions = np.append(positions, [[-1, -2, -3, -1, -2, -3, -1.5, -2.5, -1, -2, -3],
                                      [0,0,0,1,1,1, 2,2, 3,3,3],
                                      [0]*11],
                          axis=-1)
    neurons.positions = positions
    retina = neurons[0:total_retina]
    xyt = neurons[total_retina:total_retina + 3]
    sxt = neurons[total_retina + 3:total_retina + 6]
    minxyt_1 = neurons[total_retina + 6: total_retina + 8]
    greens = neurons[total_retina + 8: total_retina + 11]

    """sim.Projection(presynaptic_population=retina,
                   postsynaptic_population=retina,
                   connector=sim.DistanceDependentProbabilityConnector("d<1.5"),  # Connect all neurons in the neighborhood
                   space=sim.Space("xyz"),
                   synapse_type=sim.StaticSynapse(weight=1/8, delay=0.1),
                   receptor_type="excitatory")"""


    sim.Projection(presynaptic_population=xyt,
                   postsynaptic_population=sxt,
                   connector=sim.OneToOneConnector(),
                   synapse_type=facilitating_syn,
                   receptor_type='excitatory')

    sim.Projection(presynaptic_population=minxyt_1,
                   postsynaptic_population=sxt,
                   connector=sim.FromListConnector([(0, 0), (0, 1), (0, 2)]),
                   synapse_type=depressing_syn,
                   receptor_type='excitatory')

    sim.initialize(neurons, v=neurons.get('v_rest'))
    return neurons, total_retina

circuit, total_retina = create_brain()
retina = circuit[0:total_retina]
xyt = circuit[total_retina:total_retina + 3]
sxt = circuit[total_retina + 3:total_retina + 6]
minxyt_1 = circuit[total_retina + 6:total_retina + 8]
greens = circuit[total_retina + 8:total_retina + 11]
