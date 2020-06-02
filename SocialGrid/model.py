# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers as KL
from tensorflow.python.keras import models as KM
from sklearn.metrics import mean_absolute_error, mean_squared_error

from args import *


def residual_block(x, dilation_rate, nb_filters, kernel_size, activation='relu', dropout_rate=0.1, use_batch_norm=True):
    prev_x = x
    padding_number = dilation_rate*(kernel_size-1)
    x = tf.pad(x, [[0, 0], [padding_number, 0], [padding_number, 0], [0, 0]])
    x = KL.Conv2D(filters=nb_filters,
                  kernel_size=kernel_size,
                  dilation_rate=dilation_rate,
                  data_format="channels_last",
                  kernel_initializer=tf.random_normal_initializer(0, 0.1),
                  kernel_regularizer=keras.regularizers.l2(0.01))(x)
    if use_batch_norm:
      x = KL.BatchNormalization()(x)  # TODO should be WeightNorm here, but using batchNorm instead
      x = KL.PReLU()(x)
    else:
      x = KL.PReLU()(x)

    # 1x1 conv to match the shapes (channel dimension).
    prev_x = KL.Conv2D(nb_filters, 1, padding='same')(prev_x)
    res_x = KL.add([prev_x, x])
    res_x = KL.Activation(activation)(res_x)
    return res_x 

# Custom Loss Function
def get_loss(mask_value):
  masks = mask_value
  #divider = K.sum(masks)
  #divider = tf.dtypes.cast(divider, tf.float32)

  def masked_MSE(y_true, y_pred):
    truth = tf.math.multiply(y_true, masks)
    pred = tf.math.multiply(y_pred, masks)
    mask_loss = tf.square(tf.reduce_sum(truth) - tf.reduce_sum(pred))
    return mask_loss
  
  return masked_MSE


# Custom Metric Function
def get_metric(mask_value):
  masks = mask_value
  #divider = K.sum(masks)
  #divider = tf.dtypes.cast(divider, tf.float32)

  def masked_metric(y_true, y_pred):
    truth = tf.math.multiply(y_true, masks)
    pred = tf.math.multiply(y_pred, masks)
    mask_loss = tf.abs(tf.reduce_sum(truth) - tf.reduce_sum(pred))
    return mask_loss
  
  return masked_metric


# M-TCN model for multivariate Temporal Convolution Network
def M_TCN():
    # input layer
    train_x = KL.Input(shape=(seq_len, seq_len, 3), name="train_x")
    train_mask = KL.Input(shape=(seq_len, seq_len, 1), name="masks")
    #train_x = KL.Input(shape=(seq_len, seq_len, 3), name="train_x")

    # Temporal Layer 1
    h1 = tf.pad(train_x, [[0, 0], [FILTER_SIZE-1, 0], [FILTER_SIZE-1, 0], [0, 0]])
    h1 = KL.Conv2D(NUM_FILTERS, FILTER_SIZE, 
                   data_format="channels_last",
                   kernel_initializer=tf.random_normal_initializer(0, 0.01),
                   kernel_regularizer=keras.regularizers.l2(0.01))(h1)    

    h1 = KL.BatchNormalization()(h1)
    #h1 = KL.LeakyReLU(alpha=0.1)(h1)
    #h1 = KL.Dense(1, kernel_regularizer=keras.regularizers.l2(0.001))(h1)
    h1 = KL.PReLU()(h1)
    
    for i in dilations:
        h1 = residual_block(h1, i, NUM_FILTERS, 
                            FILTER_SIZE, activation='relu', 
                            dropout_rate=0, use_batch_norm=True)
    
    out = KL.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(h1)
    out = KL.Dense(1, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(out)
    
    # Custom Loss Function
    masked_mse_loss = get_loss(train_mask)
    # Custom Metric Function
    masked_mae_metric = get_metric(train_mask)

    model = KM.Model(inputs=[train_x, train_mask], outputs=out)
    #model = KM.Model(inputs=train_x, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(0.0001), 
                  loss=masked_mse_loss, 
                  metrics=[masked_mae_metric],
                  experimental_run_tf_function=False)
    return model