import hbp_nrp_cle.tf_framework as nrp
from std_msgs.msg import UInt8MultiArray, Int8
import numpy as np

@nrp.MapCSVRecorder("index_recorder", filename="index_real_estimate.csv",
                    headers=["Time", "real", "estimate"])
@nrp.MapRobotSubscriber('training_signal', Topic("/thimblerigger/training_signal", Int8))
@nrp.MapSpikeSink("estimate", nrp.map_neurons(range(0, 3), lambda i: nrp.brain.actors[i]), nrp.population_rate)
@nrp.Robot2Neuron()
def record_index_csv(t, index_recorder, training_signal, estimate):
    training_signal = training_signal.value
    if training_signal is None:
        return

    training_signal = training_signal.data
    if training_signal > 0:
        index_recorder.record_entry(t,
                                    training_signal,
                                    np.argmax(estimate.rate))

"""

@nrp.MapRobotSubscriber("grid_seen", Topic("/thimblerigger_solver/grid_activations", UInt8MultiArray))
@nrp.MapRobotSubscriber('training_signal', Topic("/thimblerigger/training_signal", Int8))
@nrp.MapSpikeSource("grid_neurons", nrp.map_neurons(range(0,9), lambda i: nrp.brain.grid[i]), nrp.fixed_frequency)
@nrp.MapSpikeSource("output", nrp.map_neurons(range(0, 3), lambda i: nrp.brain.output[i]), nrp.fixed_frequency)
@nrp.Robot2Neuron()
def feed_brain(t, grid_seen, training_signal, grid_neurons, output):
    grid = grid_seen.value
    training_signal = training_signal.value
    if grid is not None:

        grid = np.fromstring(grid.data, dtype=np.uint8).reshape((grid.layout.dim[0].size, grid.layout.dim[1].size))
        grid = grid.astype(np.float32)
        grid_neurons.rate = list(range(0,9))

    if training_signal is not None:
        clientLogger.info("training signal: {}".format(training_signal.data))
        output.rate = 0.
        if training_signal.data > 0:
            output[training_signal.data].rate = 1.
"""
