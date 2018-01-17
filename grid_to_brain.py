import hbp_nrp_cle.tf_framework as nrp
from std_msgs.msg import UInt8MultiArray, Int8
import numpy as np

@nrp.MapRobotSubscriber("grid", Topic("/thimblerigger_solver/grid_activations", UInt8MultiArray))
@nrp.MapRobotSubscriber('training_signal', Topic("/thimblerigger/training_signal", Int8))
@nrp.Robot2Neuron()
def feed_brain(t, grid, training_signal):
    grid = grid.value
    training_signal = training_signal.value
    if grid is not None:

        grid = np.fromstring(grid.data, dtype=np.uint8).reshape((grid.layout.dim[0].size, grid.layout.dim[1].size))

        clientLogger.info("Grid: {}".format(grid))

    if training_signal is not None > 0:
        clientLogger.info("training signal: {}".format(training_signal.data))
