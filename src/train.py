from keras.callbacks import TensorBoard
from .data_loader import load_data, add_noise, save_precomputed_data
from .model import build_autoencoder
import os

def train_autoencoder(epochs=20, batch_size=128, save_path='models/autoencoder.h5'):
    """
    Trains the autoencoder.
    """
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Save x_train for later use if needed (as per original code)
    save_precomputed_data(x_train, path='data/x_train.npy')
    
    print("Adding noise...")
    x_train_noisy = add_noise(x_train)
    x_test_noisy = add_noise(x_test)
    
    print("Building model...")
    autoencoder = build_autoencoder()
    
    print("Starting training...")
    # Ensure log directory exists
    os.makedirs('/tmp/tb', exist_ok=True)
    
    autoencoder.fit(x_train_noisy, x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
    
    print(f"Saving model to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    autoencoder.save(save_path)
    print("Training complete.")
