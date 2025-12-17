import numpy as np
from keras.datasets import cifar10
import os

def load_data():
    """
    Loads cifar10 data and preprocesses it.
    Returns:
        (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
    
    return (x_train, y_train), (x_test, y_test)

def add_noise(data, noise_factor=0.5):
    """
    Adds noise to the data.
    """
    noisy_data = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    noisy_data = np.clip(noisy_data, 0., 1.)
    return noisy_data

def save_precomputed_data(x_train, path='data/x_train.npy'):
    """
    Saves the training data to a file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, x_train)
