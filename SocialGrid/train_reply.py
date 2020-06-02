# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers as KL
from tensorflow.python.keras import models as KM
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np
import warnings
import pandas as pd
import json
import time
import math
import secrets
from utils import *
from args import *


# Getting original data
origin_data, masks, relative_data = pad_sequence(generating_dataset())

# Getting training and test data
training_data, test_data = get_data_new(origin_data, masks, relative_data, 200)

# Building dataset
train_dataset, valid_dataset = build_dataset(training_data, BATCH_SIZE_train, BATCH_SIZE_valid)

# Create an instance of the model
model = M_TCN()

callbacks = [keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-2,
    patience=6,
    mode='min',
    verbose=1,
    restore_best_weights=True
)]


model.fit(train_dataset, validation_data=valid_dataset, callbacks=callbacks, epochs=100, verbose=1)

# Build test set
test_x, test_y, list_of_random_tests = build_test_data(test_data, 50)

predictions, ground_truth = make_prediction(test_x, test_y)

print(predictions[:20])
print(ground_truth[:20])

results = np.array([predictions, ground_truth]).T

print("MAE: {}, RMSE: {}".format(mean_absolute_error(predictions, ground_truth),
      math.sqrt(mean_squared_error(predictions, ground_truth))))


