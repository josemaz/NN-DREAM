from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
from functools import reduce
from colorama import Fore, Back, Style
import glob, sys, os


# 0.5. TENSOR FLOW CONFIG
print(Fore.RED + "0.5. TENSOR FLOW CONFIG" + Style.RESET_ALL)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)


#  1. READ FILES
print(Fore.RED + "1. READ FILES" + Style.RESET_ALL)
dirtrain = "Data/NN_TF_02"
ftrain = dirtrain + "/train_pubmed.csv"
# ftrain = dirtrain + "/train_pubmed.csv"
print("Train file: ",ftrain)
dat = pd.read_csv(ftrain)
dat = dat.iloc[:,2:] # HACK to clean target,drug


#  2. SPLIT DATA
train = dat.sample(frac=0.9, random_state=1234)
test = dat.sample(frac=0.1, random_state=1234)
print(train.shape, test.shape)
# print(train.query('dgidb != 0')) # (y != 0) = 82
# print(test.query('dgidb != 0')) # (y != 0) = 19


#  3. PREPARE DATA AND SCALING
print(Fore.RED + "3. PREPARE DATA AND SCALING" + Style.RESET_ALL)
train_cols = len(train.columns) - 1
print(Fore.GREEN + "Number of features: ", train_cols)
scaler = StandardScaler()
# scaler = StandardScaler(with_mean=False) 
# Train dataset
train_val = train.values
scaler.fit(train_val)
# print(scaler.mean_)
train_val = scaler.transform(train_val)
Xtrain = train_val[:,:train_cols]
Ytrain = train_val[:,train_cols]
# print(len(Ytrain[Ytrain != 0]))
# Test dataset
test_val = test.values
test_val = scaler.transform(test_val)
Xtest = test_val[:,:train_cols]
Ytest = test_val[:,train_cols]
# print(len(Ytest[Ytest != 0]))
# Dataset
# dataset = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain))


#  4. BUILD MODEL
print(Fore.RED + "4. BUILD MODEL" + Style.RESET_ALL)
# def my_leaky_relu(x): # Por construir para exportar
#     # return tf.nn.leaky_relu(x, alpha=0.3)
#     return tf.nn.relu(x)
#120
neur = 1024 # Neuronas
model = tf.keras.Sequential([
	### INPUT
    tf.keras.layers.Dense(neur, activation='relu',input_shape=(train_cols,)),
    tf.keras.layers.Dropout(0.2),
    ### LAYER 1
    tf.keras.layers.Dense(neur, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    # ### LAYER 2
    # tf.keras.layers.Dense(neur, activation=my_leaky_relu),
    # tf.keras.layers.Dropout(0.2),
    ## OUTPUT
    tf.keras.layers.Dense(1)
])
model.summary()
opt = tf.keras.optimizers.Adam()
# opt = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae', 'mse'])


#  5. FITTING
print(Fore.RED + "5. FITTING" + Style.RESET_ALL)
model.fit(Xtrain, Ytrain, epochs=400, verbose = 2, batch_size = 32, validation_split = 0.1)


#  6. PREDICTIONS
print(Fore.RED + "6. PREDICTIONS" + Style.RESET_ALL)
predictions = model.predict(Xtest)
res = pd.DataFrame({'pred': predictions.flatten(), 'real': Ytest})
print(res[res.real > 0]) 

#  7. SAVING MODEL
print(Fore.RED + "7. SAVING MODEL" + Style.RESET_ALL)
if not os.path.isdir(dirtrain + '/models'):
    os.mkdir(dirtrain + '/models')
    print("Creating models directory")
ruta = dirtrain + '/models/nn-ch3m-pubmed.h5'
model.save(ruta)
print("Model size (Mb): ", os.path.getsize(ruta)/ (1024 * 1024))






