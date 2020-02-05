from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from colorama import Fore, Back, Style
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


#  1. LOADING MODEL
print(Fore.RED + "1. LOADING MODEL" + Style.RESET_ALL)
folder = "Data/NN_TF_02"
ruta = folder + '/models/nn-ch3m-pubmed.h5'
model = tf.keras.models.load_model(ruta)
# Show the model architecture
model.summary()


#  2. READING DATA
ftest = folder + "/test.csv"
dat_annot = pd.read_csv(ftest)
dat = dat_annot.iloc[:,2:] # HACK to clean target,drug


#  3. PREPARE DATA AND SCALING
print(Fore.RED + "3. PREPARE DATA AND SCALING" + Style.RESET_ALL)
features = len(dat.columns)
print(Fore.GREEN + "Number of features: ", features)
scaler = StandardScaler()
# scaler = StandardScaler(with_mean=False) 
# Train dataset
dat_val = dat.values
scaler.fit(dat_val)
# print(scaler.mean_)
dat_val = scaler.transform(dat_val)
Xtest = dat_val[:,:features]
# Ytest = dat_val[:,features]
# print(len(Ytrain[Ytrain != 0]


#  4. EVALUATE
# results = model.evaluate(Xtest,  Ytest)
# print('test loss, test acc:', results)


#  5. PREDICTIONS
print(Fore.RED + "5. PREDICTIONS" + Style.RESET_ALL)
predictions = model.predict(Xtest)
dfres = pd.DataFrame({'Y': predictions.flatten()})
dfres = (dfres*np.sqrt(scaler.var_[2])) + scaler.mean_[2] 
# print(dfres[dfres.pred > 0])
# Place the DataFrames side by side
print(dat.shape,dfres.shape)
hs = pd.concat([dat_annot, dfres], axis=1)
print(hs)
hs.to_csv(folder + "/models/predictions-pubmed-tf.csv")


