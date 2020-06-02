# -*- coding: utf-8 -*-
import numpy as np
import warnings
import pandas as pd
import json
import time
import math
import secrets

from args import *


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


def calculate_subtraction(mylist):
    new_list = [mylist[0]]
    for i in range(1, len(mylist)):
        new_list.append(mylist[i] - mylist[i-1])
    return new_list


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
        #print("Done, {}".format(index))
    
    print("Finished padding sequence...\n")
    origin_data = np.asarray(df)
    
    print("Start constructing mask matrix...\n")
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
    
    print("Start constructing relative time matrix...\n")
    
    relative_time_matrix = np.zeros(origin_data.shape)
    
    for i in range(relative_time_matrix.shape[1]):
        if zero_lists[i] is not None:
            relative_time_matrix[:, i] = calculate_relative_time(origin_data[:, i])
    relative_time_matrix = relative_time_matrix / relative_time_matrix.max(axis=0)
    return revision_data, temp_matrix, relative_time_matrix


def generating_dataset():
    # Read stream data
    '''
    main_stream = load_json(FILE_PATH+'data/james.json')  # All main stream data.
    # reply_stream = dh.load_json('james_reply.json') # All reply stream data.
    reply_stream = [load_json(FILE_PATH+"data/james/{}.json".format(i)) for i in range(len(main_stream))]
    '''
    _main_stream = load_json(FILE_PATH+'data/james.json')  # All main stream data.
    _reply_stream = [load_json(FILE_PATH+"data/james/{}.json".format(i)) for i in range(len(_main_stream))]

    reply_list = []
    main_time = [float(i) for i in _main_stream.created]
    # Get the reply time series
    for i,x in enumerate(_reply_stream):
        try:
            temp_list = [main_time[i]]
            #temp_list = []
            temp_list = temp_list + list(x.created_at)
            reply_list.append(temp_list)
        except:
            reply_list.append([main_time[i]])
            #reply_list.append([])
    
    return reply_list


def get_data(data_matrix, mask_matrix, relative_data, split_boundary, seq_len, index_file, train_start, train_end):
    # Read indices to a list
    foo = pd.read_csv(FILE_PATH+index_file, delimiter=',')
    list_indices_train = [int(i) for i in list(foo.iloc[:, 0])]
    
    training_set = data_matrix[:, 0:split_boundary]
    testing_set = data_matrix[:, split_boundary-seq_len:]

    mm = mask_matrix[:, 0:split_boundary]
    rr = relative_data[:, 0:split_boundary]
    
    # Take care of training and validation set
    train_feature = training_set[0: data_matrix.shape[0]-1]
    train_label = training_set[1: data_matrix.shape[0]]
    
    masks_temp = mm[0: data_matrix.shape[0]-1]
    relative_temp = rr[0: data_matrix.shape[0]-1]
    
    train_x_temp, train_y_temp, train_x, train_y =  [], [], [], []
    train_masks_temp, train_masks = [], []
    train_relative_temp, train_relatives = [], []
    
    for i in range(len(train_feature) - seq_len + 1):
        train_x_temp.append(train_feature[i:i+seq_len])
        train_y_temp.append(train_label[i:i+seq_len])
        train_masks_temp.append(masks_temp[i:i+seq_len])
        train_relative_temp.append(relative_temp[i:i+seq_len])
        
    for index_x in train_x_temp:
        for j in range(train_feature.shape[1] - seq_len + 1):
            train_x.append(index_x[:, j:j+seq_len])
    
    for index_y in train_y_temp:
        for k in range(train_label.shape[1] - seq_len + 1):
            train_y.append(index_y[:, k:k+seq_len])
            
    for index_m in train_masks_temp:
        for q in range(masks_temp.shape[1] - seq_len + 1):
            train_masks.append(index_m[:, q:q+seq_len])
            
    for index_n in train_relative_temp:
        for p in range(relative_temp.shape[1] - seq_len + 1):
            train_relatives.append(index_n[:, p:p+seq_len])
            
    print(len(train_x))
    train_x = [train_x[i] for i in list_indices_train]
    train_y = [train_y[i] for i in list_indices_train]
    train_masks = [train_masks[i] for i in list_indices_train]
    train_relatives = [train_relatives[i] for i in list_indices_train]
    
    # Validation dataset
    valid_x = train_x[train_end+1000:train_end+1200]
    valid_y = train_y[train_end+1000:train_end+1200]
    
    #valid_masks = [np.ones(i.shape, dtype=int) for i in valid_y]
    #valid_masks = train_masks[train_end:train_end+200]
    #valid_relatives = train_relatives[train_end:train_end+200]
    valid_masks = train_masks[train_end+1000:train_end+1200]
    valid_relatives = train_relatives[train_end+1000:train_end+1200]   

    # Training dataset
    train_x = train_x[train_start:train_end]
    train_y = train_y[train_start:train_end]
    
    train_masks = train_masks[train_start:train_end]
    train_relatives = train_relatives[train_start:train_end]

    return train_x, train_y, valid_x, valid_y, train_masks, train_relatives, valid_masks, valid_relatives


def get_data_new(data_matrix, mask_matrix, relative_data, seq_len):
    # Initialize empty lists for each channel
    training_instance, mask_instance, relative_instance, training_value = [], [], [], []
    test_instance, test_mask, test_relative, test_value = [], [], [], []

    # Getting start period
    zero_lists = zero_detection(data_matrix)
    
    ending_index = []
    for k in range(data_matrix.shape[1]):
        time_series = data_matrix[:, k]
        index_reverse = next((j for j, x in enumerate(time_series[::-1]) if x), None)
        try:
            index_end = len(time_series) - index_reverse
            ending_index.append(index_end)
        except:
            ending_index.append(None)
    
    time_duration = [ending_index[i] - zero_lists[i] if zero_lists[i] is not None else None for i in range(len(ending_index))]
    
    for i in range(199, 1100):
        if time_duration[i] != None:
            time_range = time_duration[i]
            if time_range>32:
                time_range = 32
            
            for j in range(time_range):
                training_instance.append(data_matrix[zero_lists[i]-seq_len+j:zero_lists[i]+j, i-seq_len+1:i+1])
                mask_instance.append(mask_matrix[zero_lists[i]-seq_len+j:zero_lists[i]+j, i-seq_len+1:i+1])
                relative_instance.append(relative_data[zero_lists[i]-seq_len+j:zero_lists[i]+j, i-seq_len+1:i+1])
                training_value.append(data_matrix[zero_lists[i]-seq_len+j+1:zero_lists[i]+j+1, i-seq_len+1:i+1])
        
        if i % 100 == 0:
          print("Training data, {} out of {}".format(i, data_matrix.shape[1]))
        
    non_empty_list = []
    for ntype in range(1100, 1600):
        if time_duration[ntype] != None:
            non_empty_list.append(ntype)
        
    for i in non_empty_list:
        time_range = time_duration[i]
        if time_range>32:
            time_range = 32

        test_instance_temp, test_mask_temp, test_relative_temp, test_value_temp = [], [], [], []

        for j in range(time_range):
            test_instance_temp.append(data_matrix[zero_lists[i]-seq_len+j:zero_lists[i]+j, i-seq_len+1:i+1])
            test_mask_temp.append(mask_matrix[zero_lists[i]-seq_len+j:zero_lists[i]+j, i-seq_len+1:i+1])
            test_relative_temp.append(relative_data[zero_lists[i]-seq_len+j:zero_lists[i]+j, i-seq_len+1:i+1])
            test_value_temp.append(data_matrix[zero_lists[i]-seq_len+j+1:zero_lists[i]+j+1, i-seq_len+1:i+1])
        
        test_instance.append(test_instance_temp)
        test_mask.append(test_mask_temp)
        test_relative.append(test_relative_temp)
        test_value.append(test_value_temp)

        if i % 100 == 0:
          print("Test data, {} out of {}".format(i, data_matrix.shape[1]))
        
    return (training_instance, mask_instance, relative_instance, training_value), (test_instance, test_mask, test_relative, test_value)


def build_dataset(training_data, batch_size_train, batch_size_val):
    x = transform_dataShape(training_data[0])
    train_masks = transform_dataShape(training_data[1])
    relative_masks = transform_dataShape(training_data[2])
    y = transform_dataShape(training_data[3])

    mm = np.zeros((seq_len, seq_len))
    mm[-1][-1] = 1
    mm = np.reshape(mm, mm.shape+(-1,)).astype('float32')

    for i in range(len(x)):
        x[i] = np.dstack((x[i], relative_masks[i], train_masks[i]))

    x_train, x, y_train, y = train_test_split(x, y, test_size=0.9, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
    
    train_dataset = tf.data.Dataset.from_tensor_slices(({"train_x": x_train, "masks": [mm]*len(x_train)}, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=100).batch(batch_size_train)
    print("Done, training set is ready...")

    valid_dataset = tf.data.Dataset.from_tensor_slices(({"train_x": x_val, "masks": [mm]*len(x_val)}, y_val))
    valid_dataset = valid_dataset.shuffle(buffer_size=100).batch(batch_size_val)
    print("Done, validation set is ready...")
    
    return train_dataset, valid_dataset


def transform_dataShape(input_matrix):
    for i in range(len(input_matrix)):
        input_matrix[i] = np.reshape(input_matrix[i], input_matrix[i].shape+(-1,)).astype('float32')
      
    return input_matrix


def flatten_list(myList):
    return [item for sublist in myList for item in sublist]


def build_test_data(test_data, num_to_select):
    '''
    secure_random = secrets.SystemRandom()        
    list_of_random_tests = sorted(secure_random.sample(list(range(len(test_data[0]))), num_to_select))

    x = transform_dataShape(flatten_list([test_data[0][i] for i in list_of_random_tests]))
    relative_masks = transform_dataShape(flatten_list([test_data[1][i] for i in list_of_random_tests]))
    test_masks = transform_dataShape(flatten_list([test_data[2][i] for i in list_of_random_tests]))
    y = transform_dataShape(flatten_list([test_data[3][i] for i in list_of_random_tests]))
    '''
    indices = []
    for i, x in enumerate(test_data[0]):
        if len(x)==32:
            indices.append(i)

    secure_random = secrets.SystemRandom()        
    list_of_random_tests = sorted(secure_random.sample(indices, num_to_select))

    x = transform_dataShape(flatten_list([test_data[0][i] for i in list_of_random_tests]))
    relative_masks = transform_dataShape(flatten_list([test_data[1][i] for i in list_of_random_tests]))
    test_masks = transform_dataShape(flatten_list([test_data[2][i] for i in list_of_random_tests]))
    y = transform_dataShape(flatten_list([test_data[3][i] for i in list_of_random_tests]))
    
    test_x, test_y = [], []
    for i in range(len(x)):
        test_x.append(np.dstack((x[i], relative_masks[i], test_masks[i])))
        test_y.append(y[i])

        if i % 100 == 0:
            print("Test data, {} out of {}".format(i, len(x)))
    
    test_x = tf.reshape(test_x, [len(test_x), seq_len, seq_len, input_channels])
    test_y = [i.reshape(seq_len, seq_len)[-1][-1] for i in test_y]
    
    return test_x, test_y, list_of_random_tests


def make_prediction(x_test, y_test):
    predictions = model.predict({"train_x": tf.reshape(x_test, [len(x_test), seq_len, seq_len, input_channels]), "masks": np.ones((len(x_test), seq_len, seq_len, 1))})
    predictions = [i.reshape(seq_len, seq_len)[-1][-1] for i in predictions]
    return predictions, y_test