import numpy as np
import tensorflow as tf
import datetime
from model.refinenet import RefineNet

refinenet = RefineNet(27)
refinenet.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

x_train = np.random.random((2, 224, 224, 3)).astype(np.float32)
y_train = np.random.random((2, 224, 224, 27)).astype(np.float32)

x_test = np.random.random((1, 224, 224, 3)).astype(np.float32)
y_test = np.random.random((1, 224, 224, 27)).astype(np.float32)

refinenet.fit(x=x_train,
          y=y_train,
          epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])
