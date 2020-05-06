import numpy as np
import tensorflow as tf
from model.refinenet import RefineNet

refinenet = RefineNet(27)
out = refinenet(np.random.random((1, 224, 224, 3)).astype(np.float32))
print(out.shape)

# tf.keras.utils.plot_model(out, to_file='model.png', show_shapes=False, show_layer_names=True,
#                 rankdir='TB', expand_nested=True, dpi=96)
