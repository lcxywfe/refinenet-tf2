import numpy as np
from model.refinenet import RefineNet

refinenet = RefineNet(27)
refinenet(np.random.random((1, 224, 224, 3)).astype(np.float32))
