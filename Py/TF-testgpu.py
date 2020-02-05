from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from colorama import Fore, Back, Style

print(Fore.RED + "¿TF withCUDA?: " + str(tf.test.is_built_with_cuda()) + Style.RESET_ALL)
print(Fore.RED + tf.__version__ + Style.RESET_ALL)
print(Fore.RED + "Num GPUs Available: " + 
	str(len(tf.config.experimental.list_physical_devices('GPU'))) + Style.RESET_ALL)