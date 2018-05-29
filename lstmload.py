import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model


dataset = pd.read_csv('03-File1.csv')
dataset_power_zero = dataset[dataset['Date']=='0']
dataset = dataset.drop(dataset_power_zero.index, axis=0)
values = dataset.iloc[:,2:6].values
values = values.astype('float32')

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
scaled=sc.fit_transform(values)
sc_predict=MinMaxScaler(feature_range=(0,1))
sc_predict.fit_transform(values[:,3:4])