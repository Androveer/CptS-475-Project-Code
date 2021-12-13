import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import time

start = time.time()

# Dataset came in two csv files. Concatenated the csv files together
tdg_data1 = pd.read_csv('tdg_data1.csv')
tdg_data2 = pd.read_csv('tdg_data2.csv')
tdg_data = pd.concat((tdg_data1,tdg_data2))

tdg_data.index = tdg_data["timestamp_pacific"]

# specify columns to plot
groups = [2, 3, 4, 5, 6, 7, 8, 9]
i = 1
# plot each column
values = tdg_data.values
"""plt.figure()
for group in groups:
	plt.subplot(len(groups), 1, i)
	plt.plot(values[:, group])
	plt.title(tdg_data.columns[group], y=0.5, loc='right')
	i += 1"""
# plt.show()

# Normalizing data
scaler = MinMaxScaler()
tdg_data_predictors = tdg_data[['tailrace_elev','tailrace_depth','flow_unit','flow_spill','flow_total','forebay_tdg','tailrace_tdg','barometer']]
scaler.fit(tdg_data_predictors)
normalized_tdg_data = tdg_data.copy()
normalized_tdg_data[['tailrace_elev','tailrace_depth','flow_unit','flow_spill','flow_total','forebay_tdg','tailrace_tdg','barometer']] = scaler.transform(tdg_data_predictors)

# Splitting the dataset
split_fraction = 0.715
train_split = int(split_fraction * int(normalized_tdg_data.shape[0]))
step = 6

past = 720
future = 72
learning_rate = 0.001
batch_size = 256
epochs = 10

date_time_key = "timestamp_pacific"

selected_features = [normalized_tdg_data.columns[i] for i in [2, 3, 4, 5, 6, 7, 8, 9]]
features = normalized_tdg_data[selected_features]
features.index = normalized_tdg_data[date_time_key]
# print(features.head())

train_data = features.iloc[0 : train_split - 1]
val_data = features.iloc[train_split:]
print(normalized_tdg_data)

print(train_data)
x_train = (train_data.drop(['tailrace_tdg'],axis=1)).to_numpy()
y_train = (train_data['tailrace_tdg']).to_numpy()

print(val_data)
x_test = (val_data.drop(['tailrace_tdg'],axis=1)).to_numpy()
y_test = (val_data['tailrace_tdg']).to_numpy()

model = keras.Sequential()
model.add(keras.Input(shape=(7,)))
model.add(layers.Dense(14, activation="relu"))
model.add(layers.Dense(1, activation="relu"))

model.summary()

model.compile(optimizer="Adam",loss="mse",metrics=[tf.metrics.MeanAbsoluteError()])
history = model.fit(x_train,y_train,batch_size=64,epochs=100)

test_loss = model.evaluate(x_test,y_test)
# print("Test loss", test_loss)

# make a prediction
ynew = model.predict(x_test)
# show the inputs and predicted outputs
# for i in range(10):
#	print("X=%s, Predicted=%s, Actual=%s" % (x_test[i], ynew[i], y_test[i]))

print(ynew)
print(y_test)


# Plot mean squared error
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['mean_absolute_error'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()


plt.figure(figsize=(10,6))
plt.plot(y_test, color='blue', label='Actual Tailrace TDG')
plt.plot(ynew , color='red', label='Predicted Tailrace TDG')
plt.title('Tailrace TDG Prediction')
plt.xlabel('Time Intervals')
plt.ylabel('Tailrace TDG')
plt.legend()
plt.show()


end = time.time()

print("Executed in " + str(end - start) + " seconds")