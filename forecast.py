import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import History, EarlyStopping


class CustomMinMaxScaler:
    def __init__(self, x):
        self.min_ = x.min()
        self.max_ = x.max()

    def scale(self, x):
        return (x - self.min_) / (self.max_ - self.min_)

    def inverse_scale(self, x):
        return x * (self.max_ - self.min_) + self.min_


argparser = argparse.ArgumentParser()
argparser.add_argument('-d', help='Dataset', default='nasdaq2007_17.csv')
argparser.add_argument('-n', help='Number of timeseries', type=int, default=1)
args = argparser.parse_args()

dataset = args.d
num_timeseries = args.n

# Hyperparameters
test_percent = 0.2
delay = 200
max_epochs = 20
batch_size = 32
lstm_units = 8
lstm_layers = 1
dropout = 0.1

experiment_folder = 'experiments'
os.makedirs(experiment_folder, exist_ok=True)

# Read dataset
df = pd.read_csv(dataset, header=None, sep="\t")
df = df.drop(columns=[0])
data = df.head(int(num_timeseries)).values

num_columns = len(df.columns)
test_size = int(num_columns * test_percent)
train_size = num_columns - test_size
train_set = data[:, 0:train_size]
# Afairoume to delay kathws xreiazomaste tis times gia na provlepsoume thn prwth timh tou test set
test_set = data[:, train_size - delay:num_columns]
print(f'Train data: {train_set.shape} - Test data: {test_set.shape}')

# Min-Max scaler wste na metatrepsoume oles tis times sto [0, 1]
scaler = CustomMinMaxScaler(train_set)
train_set_scaled = scaler.scale(train_set)
test_set_scaled = scaler.scale(test_set)

# ! Run on all timeseries at the same time
# Training data
trainX = []
trainY = []
for i in range(len(data)):
    for j in range(delay, train_size):
        trainX.append(train_set_scaled[i, j - delay:j])
        trainY.append(train_set_scaled[i, j])
trainX = np.array(trainX)
trainY = np.array(trainY)
trainX = np.expand_dims(trainX, -1)
# Test data
testX = []
testY = []
for i in range(len(data)):
    testX.append([])
    testY.append([])
    for j in range(delay, test_size + delay):
        testX[i].append(test_set_scaled[i, j - delay:j])
        testY[i].append(test_set_scaled[i, j])
    testX[i] = np.array(testX[i])
    testY[i] = np.array(testY[i])
    testX[i] = np.expand_dims(testX[i], -1)

print('Neural network input shape')
print(f'Train shape: {trainX.shape}')

model = Sequential()
if lstm_layers == 1:
    model.add(LSTM(units=lstm_units, input_shape=(trainX.shape[1], 1)))
    model.add(Dropout(dropout))
else:
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(trainX.shape[1], 1)))
    model.add(Dropout(dropout))
    for i in range(lstm_layers - 2):
        model.add(LSTM(units=lstm_units, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout))
model.add(Dense(units=1))
print(model.summary())

# Training
model.compile(optimizer='adam', loss='mean_squared_error')
history = History()
# Epilegoume tyxaia 10% tou training set ws validation, gia na mporoume na apofygoume to overfitting
# Early stopping, dhladh an se 3 epoxes den exei veltiwthei to validation loss, stamatame to training
early_stop = EarlyStopping(patience=3, verbose=1)
hist = model.fit(trainX, trainY,
                 epochs=max_epochs,
                 batch_size=batch_size,
                 validation_split=0.1,
                 callbacks=[history, early_stop])
hist = hist.history

plt.figure()
plt.plot(hist['loss'], color='red', label='Train loss')
plt.plot(hist['val_loss'], color='blue', label='Validation loss')
plt.savefig(os.path.join(experiment_folder, 'training_history.png'))

test_loss_file = open(os.path.join(experiment_folder, 'test_losses.txt'), 'w')
for i in range(len(testX)):
    test_loss = model.evaluate(testX[i], testY[i])
    print(f'Test sample {i} loss: {test_loss}')
    test_loss_file.write(f'Test sample {i} loss: {test_loss}\n')
    predicted = model.predict(testX[i])
    predicted = scaler.inverse_scale(predicted)
    predicted = np.squeeze(predicted)
    plt.figure()
    plt.plot(test_set[i, delay:], color='red', label='Real Price')
    plt.plot(predicted, color='blue', label='Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(os.path.join(experiment_folder, f'test_{i}'))
test_loss_file.close()
