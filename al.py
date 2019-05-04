import tensorflow as tf
import numpy as np
from scipy.special import entr
from keras import backend as K
import pickle
import os


def build(network_size):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(network_size, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def al():
    dataset = tf.keras.datasets.fashion_mnist
    pool = 60000
    init_batch = 200
    query_batch = 200
    iteration_num = 100
    
    network_size = 32
    epochs = 20
    
    aquisition = "entropy"
    stride = 1
    
    
    out = "./history/"
    if not os.path.exists(out):
        os.makedirs(out)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    K.tensorflow_backend.set_session(tf.Session(config=config))
    
    # init
    (x_train, y_train),(x_test, y_test) = dataset.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    permute = np.random.permutation(len(x_train))
    x_train = x_train[permute]
    y_train = y_train[permute]
    x_train = x_train[:pool]
    y_train = y_train[:pool]
    
    print(np.shape(x_train), np.shape(y_train))
    
    labelled_x = x_train[:init_batch]
    labelled_y = y_train[:init_batch]
    unlabelled_x = x_train[init_batch:]
    unlabelled_y = y_train[init_batch:]
    
    for iter in range(iteration_num):
        print("iteration ", iter)
        print(np.shape(labelled_x), np.shape(unlabelled_x))
        
        # train
        model = build(network_size)
        history = model.fit(labelled_x, labelled_y, 
            epochs=epochs, validation_data=(x_test, y_test), verbose=0)
        pickle.dump( history.history, open(out + "history_" + str(iter), "wb" ) )
        metric = model.evaluate(x_test, y_test, verbose=0)
        print(model.metrics_names, metric)
        
        # query
        unlabelled_size = np.shape(unlabelled_x)[0]
        prediction = model.predict(unlabelled_x)
        entropy = entr(prediction).sum(axis=1) / np.log(2)
        if (aquisition == "random"):
            acq = np.random.rand(unlabelled_size)
        elif (aquisition == "entropy"):
            acq = entropy
        else:
            print("error!")
        
        arg = np.argsort(acq)[::-1]
        sorted_x = unlabelled_x[arg]
        sorted_y = unlabelled_y[arg]
        
        # diversity
        sorted_x = np.append(sorted_x[np.mod(np.arange(unlabelled_size),stride)==0], 
            sorted_x[np.mod(np.arange(unlabelled_size),stride)!=0], axis=0)
        sorted_y = np.append(sorted_y[np.mod(np.arange(unlabelled_size),stride)==0], 
            sorted_y[np.mod(np.arange(unlabelled_size),stride)!=0], axis=0)
        
        # label
        labelled_x = np.append(labelled_x, sorted_x[:query_batch], axis=0)
        labelled_y = np.append(labelled_y, sorted_y[:query_batch], axis=0)
        unlabelled_x = sorted_x[query_batch:]
        unlabelled_y = sorted_y[query_batch:]
        

al()



