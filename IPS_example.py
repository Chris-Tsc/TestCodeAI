# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Modified model for IPS: Binary classification (Block/Allow)
def model_aps(lr=1e-4, N=64, inshape=None):
    model = Sequential()
    model.add(Conv1D(filters=N, kernel_size=3, activation='relu', input_shape=(inshape, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary output (0=Allow, 1=Block)

    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Action mechanism to block traffic
def take_action(prediction, source_ip):
    if prediction == 1:  # Attack detected
        print(f"Blocking IP address {source_ip} due to detected attack!")
        os.system(f"iptables -A INPUT -s {source_ip} -j DROP")
    else:
        print(f"Traffic from {source_ip} is normal, allowing connection.")

# Real-time prediction based on model output
def predict_and_act(model, packet_data, source_ip):
    packet_data = np.reshape(packet_data, (1, packet_data.shape[0], 1))  # Reshaping for Conv1D
    prediction = model.predict(packet_data)  # 0 = Allow, 1 = Block
    take_action(int(prediction > 0.5), source_ip)

# Training and evaluation
def loadDataset():
    filename = 'https://raw.githubusercontent.com/kdemertzis/EKPA/main/Data/pcap_data.csv'
    trainfile = pd.read_csv(filename)
    data = trainfile.to_numpy()
    data = data[data[:, 25] != 'DrDoS_LDAP']  # Filter out 'DrDoS_LDAP'
    np.random.shuffle(data)

    label = data[:, 25]
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)

    # Map to binary labels (0 = normal, 1 = attack)
    label = np.where(label > 0, 1, 0)

    inx_sel = -1 + np.array([11, 9, 7, 10, 1, 4, 17, 19, 21,
                             18, 22, 24, 23, 12, 25, 20])

    data = data[:, inx_sel]
    dmin = data.min(axis=0)
    dmax = data.max(axis=0)
    data = (data - dmin) / (dmax - dmin)

    train_data, test_data, train_label, test_label = \
        train_test_split(data, label, test_size=0.20, stratify=label)

    train_data, val_data, train_label, val_label = \
        train_test_split(train_data, train_label, test_size=0.125, stratify=train_label)

    return train_data.astype('float32'), train_label.astype('int32'), \
           val_data.astype('float32'), val_label.astype('int32'), \
           test_data.astype('float32'), test_label.astype('int32')

# Define nclass for binary classification (Block/Allow)
nclass = 2

train_data, train_labelp, val_data, val_labelp, test_data, test_labelp = loadDataset()

# Training the model
model = model_aps(lr=1e-4, N=64, inshape=train_data.shape[1])
history = model.fit(train_data,
                    train_labelp,
                    epochs=100,
                    batch_size=256,
                    validation_data=(val_data, val_labelp))

# Real-time prediction simulation loop
for i in range(len(test_data)):
    packet_data = test_data[i]
    source_ip = "100.100.1.1"  # Example source IP (this would be dynamic in a real system)
    predict_and_act(model, packet_data, source_ip)