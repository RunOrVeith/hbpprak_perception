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
    facilitating_syn = sim.TsodyksMarkramSynapse(U=0.5, tau_rec=20.0,
                                                 tau_facil=100.0, weight=0.1,
                                                 delay=0.1)

    cell_class = sim.IF_cond_exp(**SENSORPARAMS)

    neurons = sim.Population(6, cellclass=cell_class)


    sim.initialize(neurons, v=neurons.get('v_rest'))
    return neurons

circuit = create_brain()
greens = circuit[:3]
prediction = circuit[3:]
