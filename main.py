import argparse
import numpy as np
import os
from keras.models import load_model, Model
from src.data_loader import load_data, add_noise
from src.train import train_autoencoder
from src.evaluate import compute_average_precision_score
from src.visualization import retrieve_closest_images, plot_denoised_images

def load_encoder(model_path='models/autoencoder.h5'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    autoencoder = load_model(model_path)
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
    return autoencoder, encoder

def main():
    parser = argparse.ArgumentParser(description="CBIR Autoencoder Project")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the autoencoder')
    train_parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    train_parser.add_argument('--save_path', type=str, default='models/autoencoder.h5', help='Path to save the model')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--n_test_samples', type=int, default=1000, help='Number of test samples')
    eval_parser.add_argument('--n_train_samples', type=int, default=1000, help='Number of training samples to compare against')
    
    # Retrieval Demo command
    demo_parser = subparsers.add_parser('demo', help='Run image retrieval demo')
    demo_parser.add_argument('--index', type=int, default=1, help='Index of test image to use')
    demo_parser.add_argument('--n_samples', type=int, default=10, help='Number of similar images to retrieve')
    
    # Denoise Demo command
    denoise_parser = subparsers.add_parser('denoise', help='Run denoising demo')

    args = parser.parse_args()

    if args.command == 'train':
        train_autoencoder(epochs=args.epochs, batch_size=args.batch_size, save_path=args.save_path)

    elif args.command == 'evaluate':
        print("Loading data and model...")
        (x_train, y_train), (x_test, y_test) = load_data()
        
        # Load model
        model_path = 'models/autoencoder.h5'
        _, encoder = load_encoder(model_path)

        x_train = x_train[:args.n_train_samples] # Limit training samples for speed if needed? 
        # Actually logic expects all learned codes, let's keep consistent with original
        # Original: n_train_samples was used to slice correct? No, it passed n_train_samples to compute_average_precision_score
        # which sliced the output results? No, compute_average_precision_score takes n_samples as last arg
        # to slice the output of retrieve_closest_elements.
        
        # Re-reading original `test_model.py`:
        # score = compute_average_precision_score(test_codes[indexes], y_test[indexes], learned_codes, n_train_samples)
        # It predicts ALL x_train to get learned_codes.
        # But inside compute_average_precision_score -> retrieve_closest_elements -> it returns sorted list
        # And then valid slice [:n_samples] (n_samples is passed as n_train_samples).
        # So essentially "Precision at K" where K=n_train_samples.
        
        learned_codes = encoder.predict(x_train)
        learned_codes = learned_codes.reshape(learned_codes.shape[0], -1)
        
        test_codes = encoder.predict(x_test)
        test_codes = test_codes.reshape(test_codes.shape[0], -1)
        
        # Select random subset if n_test_samples < len(x_test)
        indexes = np.arange(len(y_test))
        # np.random.shuffle(indexes) # Optional, keep consistent
        indexes = indexes[:args.n_test_samples]
        
        print(f'Computing score for top {args.n_train_samples} retrieved items...')
        score = compute_average_precision_score(test_codes[indexes], y_test[indexes], learned_codes, args.n_train_samples, y_train)
        print(f'Model score (Mean Average Precision): {score}')

    elif args.command == 'demo':
        print("Running retrieval demo...")
        (x_train, y_train), (x_test, y_test) = load_data()
        _, encoder = load_encoder()
        
        idx = args.index
        # Ensure idx is within bounds
        if idx >= len(x_test):
            idx = 0
            print("Index out of bounds, using 0.")
            
        retrieve_closest_images(x_test[idx], y_test[idx], encoder, x_train, y_train, n_samples=args.n_samples)

    elif args.command == 'denoise':
        print("Running denoising demo...")
        (x_train, y_train), (x_test, y_test) = load_data()
        autoencoder, _ = load_encoder()
        x_test_noisy = add_noise(x_test)
        plot_denoised_images(autoencoder, x_test_noisy)

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
