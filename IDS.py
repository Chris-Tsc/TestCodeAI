# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

def model_conv1D(lr=1e-4, N=64, inshape=None):
    model = Sequential()
    model.add(Conv1D(filters=N, kernel_size=3, activation='relu', input_shape=(inshape, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(12, activation='softmax'))  # Assuming 12 classes for classification

    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_dense(lr=1e-4, N=64, inshape=None):
    model = Sequential()
    model.add(Dense(N, input_dim=inshape, activation='relu'))
    model.add(Dense(N, activation='relu'))
    model.add(Dense(12, activation='softmax'))  # Assuming 12 classes

    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_lstm(lr=1e-4, N=64, inshape=None):
    model = Sequential()
    model.add(LSTM(N, input_shape=(inshape, 1), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(12, activation='softmax'))  # Assuming 12 classes

    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


epochs = 100
nclass = 12

def loadDataset():
    # Put dataset path here !
    filename = 'https://raw.githubusercontent.com/kdemertzis/EKPA/main/Data/pcap_data.csv'

    trainfile = pd.read_csv(filename)
    data = pd.DataFrame(trainfile).to_numpy()
    data = data[data[:, 25] != 'DrDoS_LDAP']  # Filter out 'DrDoS_LDAP'
    np.random.shuffle(data)

    label = data[:, 25]

    # Use LabelEncoder to convert categorical labels to integers
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)

    # SELECT FEATURES ----------------------------------------------------
    inx_sel = -1 + np.array([11, 9, 7, 10, 1, 4, 17, 19, 21,
                             18, 22, 24, 23, 12, 25, 20])

    # MIN-MAX normalization
    data = data[:, inx_sel]
    dmin = data.min(axis=0)
    dmax = data.max(axis=0)
    data = (data - dmin) / (dmax - dmin)

    # Split data
    train_data, test_data, train_label, test_label = \
        train_test_split(data, label, test_size=0.20, stratify=label)

    # Train 70%, Validation 10%
    train_data, val_data, train_label, val_label = \
        train_test_split(train_data, train_label, test_size=0.125, stratify=train_label)

    return train_data.astype('float32'), train_label.astype('int32'), \
        val_data.astype('float32'), val_label.astype('int32'), \
        test_data.astype('float32'), test_label.astype('int32')

# -- LOAD DATA -----------------------------------------------------------------
train_data, train_labelp, val_data, val_labelp, test_data, test_labelp = loadDataset()

# Convert labels to categorical (using train_labelp, val_labelp, test_labelp)
train_label = to_categorical(train_labelp, nclass)
val_label = to_categorical(val_labelp, nclass)
test_label = to_categorical(test_labelp, nclass)

print('train_data.shape=', train_data.shape)
print('test_data.shape=', test_data.shape)
print('val_data.shape=', val_data.shape)  # Fixed typo here

#get the number of features
inshape = train_data.shape[1]

# Class balancing weights
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(train_labelp),
                                                  y=train_labelp)

class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# -- CALLBACKS -----------------------------------------------------------------
earlyStopping = EarlyStopping(monitor='val_loss',
                              patience=30,
                              verbose=0,
                              mode='min')

modelCheckPoint = ModelCheckpoint('./savemodels/model5class.weights.{epoch:03d}-{val_accuracy:.4f}.keras',
                                  save_best_only=True,
                                  monitor='val_accuracy',
                                  mode='max')

# -- Baseline models-----------------------------------------------------------
# -- Conv1d
model = model_conv1D(lr=1e-4, N=64, inshape=inshape)
# -- Dense
# model = model_dense(lr=1e-4, N=64, inshape=inshape)
# -- LSTM
# model = model_lstm(lr=1e-4, N=64, inshape=inshape)

model.summary()
# -----------------------------------------------------------------------------
# print model to an image file
# dot_img_file = 'model1.png'
# plot_model(model, to_file=dot_img_file, show_shapes=True)

# -- TRAIN MODEL --------------------------------------------------------------
history = model.fit(train_data,
                    train_label,
                    shuffle=True,
                    epochs=epochs,
                    batch_size=256,  # 256, #128, #32, 64
                    validation_data=(val_data, val_label),
                    callbacks=[modelCheckPoint],
                    class_weight=class_weights)

# -- Load best model ----------------------------------------------------------
str_models = os.listdir('./savemodels')
str_models = np.sort(str_models)
best_model = str_models[str_models.size-1]
print('best_model=', best_model)
model.load_weights('./savemodels/'+best_model)

# --Confusion matrix ----------------------------------------------------------
print('TEST DATA-Confusion matrix:')
pred = model.predict(test_data)
pred_y = pred.argmax(axis=-1)

cm = confusion_matrix(test_labelp.astype('int32'), pred_y)
print(cm)

print('Accuracy ratios for each class')
print('Class 1      =', cm[0, 0]/np.sum(cm[0, :]))
print('Class 2      =', cm[1, 1]/np.sum(cm[1, :]))
print('Class 3      =', cm[2, 2]/np.sum(cm[2, :]))



# -- Confusion matrix plot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
label=np.array(["Class 1","Class 2","Class 3"])

cmo = ConfusionMatrixDisplay(cm,display_labels=label)
fig, ax = plt.subplots(figsize=(3,3))
cmo.plot(ax=ax, xticks_rotation=45)


# Plot training and validation accurry and loss graphs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

np.save('historydata.npy',[acc,val_acc,loss,val_loss])
[acc, val_acc, loss, val_loss] = np.load('historydata.npy')

plt.figure()
epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r.', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r.', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
