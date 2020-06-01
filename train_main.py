# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:25:25 2020

@author: cling
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow.keras as keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers as KL
from tensorflow.python.keras import models as KM
from tensorflow.keras import Sequential
import numpy as np
import warnings
import pandas as pd
import json, os
import time
import math
import secrets

from args import *


# M-TCN model for multivariate Temporal Convolution Network
def MTCN_thread():
    # input layer
    train_x = KL.Input(shape=(200, 200, input_channels), name="train_x")
    
    # Temporal Layer 1
    h1 = tf.pad(train_x, [[0, 0], [4, 0], [4, 0], [0, 0]])
    h1 = KL.Conv2D(32, FILTER_SIZE, 
                   input_shape=(200, 200, input_channels), 
                   data_format="channels_last",
                   kernel_initializer=tf.random_normal_initializer(0, 0.1),
                   kernel_regularizer=keras.regularizers.l2(0.01))(h1)    
    #h1 = KL.BatchNormalization()(h1)
    #h1 = KL.LeakyReLU(alpha=0.1)(h1)
    h1 = KL.Dense(1, kernel_regularizer=keras.regularizers.l2(0.01))(h1)
    h1 = KL.PReLU()(h1)
    
    # Temporal Layer 2
    h2 = tf.pad(h1, [[0, 0], [8, 0], [8, 0], [0, 0]])
    h2 = KL.Conv2D(32, FILTER_SIZE, 
                   dilation_rate=2, 
                   data_format="channels_last",
                   kernel_initializer=tf.random_normal_initializer(0, 0.01),
                   kernel_regularizer=keras.regularizers.l2(0.01))(h2)    
    #h2 = KL.BatchNormalization()(h2)
    #h1 = KL.LeakyReLU(alpha=0.1)(h1)
    h2 = KL.Dense(1, kernel_regularizer=keras.regularizers.l2(0.01))(h2)
    h2 = KL.PReLU()(h2)
    
    # Temporal Layer 3
    h3 = tf.pad(h2, [[0, 0], [16, 0], [16, 0], [0, 0]])
    h3 = KL.Conv2D(32, FILTER_SIZE, 
                   dilation_rate=4, 
                   data_format="channels_last",
                   kernel_initializer=tf.random_normal_initializer(0, 0.01),
                   kernel_regularizer=keras.regularizers.l2(0.01))(h3)    
    #h3 = KL.BatchNormalization()(h3)
    #h1 = KL.LeakyReLU(alpha=0.1)(h1)
    h3 = KL.Dense(1, kernel_regularizer=keras.regularizers.l2(0.01))(h3)
    h3 = KL.PReLU()(h3)

    h3 = KL.Flatten()(h3)
    h3 = KL.Dense(16, activation='relu', use_bias=True)(h3)
    h3 = KL.Dense(1, activation='relu', use_bias=True)(h3)

    model = KM.Model(inputs=train_x, outputs=h3)
    model.compile(optimizer=keras.optimizers.Adam(0.0001), 
                  loss=keras.losses.mean_squared_error, 
                  metrics=['mae'],
                  experimental_run_tf_function=False)
    return model


def load_json(filename):
    """Load json files in a given directory"""
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))

    data = pd.DataFrame(data)
    return data


def count_number(test_list, l, r): 
  c = 0 
  # traverse in the list1 
  for x in test_list: 
    # condition check 
    if x>= l and x<= r: 
      c+= 1 
  return c


def zero_detection(myList):
    list_index = list()
    for i in range(myList.shape[1]):
        col_vec = myList[:, i].tolist()
        try:
            index = next((j for j, x in enumerate(col_vec) if x), None)
            list_index.append(index)
        except:
            list_index.append(None)
    return list_index


def calculate_relative_time(time_series):
    '''
    index_start = next((j for j, x in enumerate(time_series) if x), None)
    index_reverse = next((j for j, x in enumerate(time_series[::-1]) if x), None)
    index_end = len(time_series) - index_reverse - 1
    
    relative_time_series = np.zeros(len(time_series))
    relative_time_series[index_start:index_end] = np.array(range(index_end-index_start))+1
    '''
    index_start = next((j for j, x in enumerate(time_series) if x), None)
    relative_time_series = np.zeros(len(time_series))
    relative_time_series[index_start:] = np.array(range(len(time_series)-index_start))+1
    
    return relative_time_series


def pad_sequence(reply_list):
    earliest_time = int(reply_list[0][0])
    latest_time = int(reply_list[-1][-1])
    
    # Get the time duration based on the earlist and latest time
    time_duration = []
    for i in range(earliest_time, latest_time, DURATION):
        time_duration.append(i)
    
    time_duration_mod = []
    for i in range(len(time_duration)-1):
        time_duration_mod.append((time_duration[i], time_duration[i+1]))
    
    # Generate a dataframe storing all the sequence
    df = pd.DataFrame([])
    
    for index, seq in enumerate(reply_list):
        time_series = []
        for k in time_duration_mod:
            time_series.append(count_number(seq, k[0], k[1]))
    
        df['{}'.format(index)] = time_series
        if index%100 == 0: 
          print("Done, {}".format(index))
    
    print("Finished padding sequence...\n")
    origin_data = np.asarray(df)
    
    print("Start constructing relative time matrix...\n")

    revision_data = np.copy(origin_data)
    
    for i in range(revision_data.shape[1]):
        index = (revision_data[:, i]!=0).argmax(axis=0)
        revision_data[:, i][index] -= 1
    
    zero_lists = zero_detection(origin_data)
    temp_matrix = np.zeros(origin_data.shape)
    
    for i in range(temp_matrix.shape[1]):
        if zero_lists[i] is not None:
            temp_1 = np.zeros(temp_matrix.shape[0])[:zero_lists[i]]
            temp_2 = np.zeros(temp_matrix.shape[0])[zero_lists[i]:]+1
            temp = np.concatenate([temp_1, temp_2])
            temp_matrix[:, i] = temp
        else:
            temp_matrix[:, i] = np.zeros(temp_matrix.shape[0])
      
    relative_time_matrix = np.zeros(origin_data.shape)
    
    for i in range(relative_time_matrix.shape[1]):
        if zero_lists[i] is not None:
            relative_time_matrix[:, i] = calculate_relative_time(origin_data[:, i])
    
    relative_time_matrix = relative_time_matrix / relative_time_matrix.max(axis=0)

    return origin_data, temp_matrix, relative_time_matrix


def generating_dataset():
    # Read stream data
    '''
    main_stream = load_json(FILE_PATH+'data/james.json')  # All main stream data.
    # reply_stream = dh.load_json('james_reply.json') # All reply stream data.
    reply_stream = [load_json(FILE_PATH+"data/james/{}.json".format(i)) for i in range(len(main_stream))]
    '''
    _reply_stream = []
    print('Start Reading...')
    _main_stream = load_json(FILE_PATH+'data/james.json')  # All main stream data.
    for i in range(len(_main_stream)):
      _reply_stream.append(load_json(FILE_PATH+"data/james/{}.json".format(i)))
      print("Done, {}".format(i))

    print('Start Making list...')
    reply_list = []
    main_time = [float(i) for i in _main_stream.created]
    # Get the reply time series
    for i,x in enumerate(_reply_stream):
        try:
            temp_list = [main_time[i]]
            temp_list = temp_list + list(x.created_at)
            reply_list.append(temp_list)
        except:
            reply_list.append([main_time[i]])
    
    return reply_list


def get_data(origin_data, relative_data, mask_data, seq_len):
    features, features_relative, features_mask, labels_temp, labels = [], [], [], [], []
    index_list = zero_detection(relative_data)
    truncated_index = index_list[seq_len-1:1601]
    
    for i, x in enumerate(truncated_index):
        features.append(origin_data[x-seq_len+1:x+1, i:i+seq_len])
        features_mask.append(mask_data[x-seq_len+1:x+1, i:i+seq_len])
        features_relative.append(relative_data[x-seq_len+1:x+1, i:i+seq_len])
        labels_temp.append(origin_data[x:, i+seq_len])
        #labels_temp.append(origin_data[x:x+label_len, i+seq_len])
    '''
    zero_indices = []
    for i, x in enumerate(labels_temp):
        if np.count_nonzero(x) == 0:
            zero_indices.append(i)
    
    features = [v for i,v in enumerate(features) if i not in zero_indices]
    labels_temp = [v for i,v in enumerate(labels_temp) if i not in zero_indices]
    '''
    
    for i in labels_temp:
        labels.append(next((j for j, x in enumerate(i) if x), None))
    
    print('Getting dataset ready...')
    
    for i in range(len(features)):
        #features[i] = np.dstack((features[i], features_relative[i], features_mask[i]))
        features[i] = np.dstack((features[i], features_relative[i]))

    test_x = features[1100:1400]
    test_y = labels[1100:1400]

    features = features[:1100]
    labels = labels[:1100]

    return features, labels, test_x, test_y


def transform_dataShape(input_matrix):
    for i in range(len(input_matrix)):
        input_matrix[i] = np.reshape(input_matrix[i], input_matrix[i].shape+(-1,)).astype('float32')
      
    return input_matrix


def build_tf_dataset(x, y, train_batch_size, valid_batch_size):
    '''
    x_train = x[:train_valid_split]
    y_train = y[:train_valid_split]
    
    x_valid = x[train_valid_split:]
    y_valid = y[train_valid_split:]
    '''
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1)

    print('Building Training Dataset')
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=100).batch(train_batch_size)

    print("Building Validation Dataset")
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    valid_dataset = valid_dataset.shuffle(buffer_size=100).batch(valid_batch_size)
    
    return train_dataset, valid_dataset


# Getting original data
origin_data, masks, relative_data = pad_sequence(generating_dataset())

features, labels, test_x, test_y = get_data(origin_data, relative_data, masks, SEQ_LEN_MAIN)

'''
new_features, new_labels = [], []
for i, x in enumerate(features):
  if x.shape[0] != 0:
    new_features.append(x)
    new_labels.append(labels[i])
'''

train_dataset, valid_dataset = build_tf_dataset(new_features, new_labels, BATCH_SIZE_train, BATCH_SIZE_valid)


# Create an instance of the model
model = MTCN_thread()

# Initialize the Callback
callbacks = [keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-1,
    patience=10,
    mode='min',
    verbose=1,
    restore_best_weights=True
)]

model.fit(train_dataset, validation_data=valid_dataset, epochs=50, verbose=1)

os.makedirs(f"models/", exist_ok=True)
model.save('models/{}.h5'.format('james'))


secure_random = secrets.SystemRandom()        
list_of_random_tests = sorted(secure_random.sample(list(range(len(test_x))), 100))

testing = [test_x[i] for i in list_of_random_tests]
ground_truth = [test_y[i] for i in list_of_random_tests]

testing = tf.reshape(testing, [len(testing), SEQ_LEN_MAIN, SEQ_LEN_MAIN, 2])

test_x_pred = model.predict(testing).flatten().tolist()

# Print out all the error results
print(mean_absolute_error(test_y[:20], test_x_pred[:20]), math.sqrt(mean_squared_error(test_y[:20], test_x_pred[:20])))
print(mean_absolute_error(test_y[:40], test_x_pred[:40]), math.sqrt(mean_squared_error(test_y[:40], test_x_pred[:40])))
print(mean_absolute_error(test_y[:60], test_x_pred[:60]), math.sqrt(mean_squared_error(test_y[:60], test_x_pred[:60])))
print(mean_absolute_error(test_y[:80], test_x_pred[:80]), math.sqrt(mean_squared_error(test_y[:80], test_x_pred[:80])))
print(mean_absolute_error(test_y[:100], test_x_pred[:100]), math.sqrt(mean_squared_error(test_y[:100], test_x_pred[:100])))

