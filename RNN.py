# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:50:03 2023

@author: aitza
"""



import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# Import dataset
dataset = pd.read_excel('D:\Extra\Modulation_Classification\Features.xlsx', sheet_name='1')

# Handling missing values
dataset.dropna(inplace=True)

# Separating the features and labels
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Reshape data for LSTM input (assuming X has 2D shape)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the RNN model
model = Sequential()
model.add(LSTM(units=64, input_shape=(1, X_train.shape[2]), return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(units=64))
model.add(BatchNormalization())
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'],
              run_eagerly=None)

print(model.summary())

# Define the checkpoint callback to save the model with the minimum loss
checkpoint_callback = ModelCheckpoint('model_min_loss.h5', 
                                      monitor='val_loss', 
                                      save_best_only=True)

# Define the CSVLogger callback to log training metrics to a CSV file
csv_logger = CSVLogger('training_log.csv')

# Train the model with the checkpoint callback
model.fit((X_train , y_train),
          batch_size=32,
          epochs=100, 
          validation_data=(X_val , y_val), 
          callbacks=[checkpoint_callback, csv_logger])



# Load the saved model
model = tf.keras.models.load_model('model_min_loss.h5')

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate((X_test , y_test))
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)


# Generate predictions for the test dataset
predictions = model.predict((X_test , y_test))
predicted_labels = tf.argmax(predictions, axis=1)

class_names = label_encoder.classes_

# Print classification report
print('\nClassification Report:')
print(classification_report(y_test, predicted_labels, target_names=class_names))

# Print confusion matrix
print('\nConfusion Matrix:')
cm = confusion_matrix(y_test, predicted_labels)
print(cm)

# Calculate and print AUC score
auc_score = roc_auc_score(y_test, predicted_labels, multi_class='ovr')
print('\nAUC Score:', auc_score)