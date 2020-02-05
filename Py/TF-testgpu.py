from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf

print("¿TF withCUDA?: ",tf.test.is_built_with_cuda())
print(tf.__version__)
print(tf.config.experimental.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))