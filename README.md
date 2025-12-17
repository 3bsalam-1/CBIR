# Content-Based Image Retrieval (CBIR) using Autoencoders

![Project Banner](output_images/retrieved_results_1.jpg) <!-- Replace with a representative image if available, using generated results -->

A Deep Learning project that implements Content-Based Image Retrieval (CBIR) functionality using a Convolutional Autoencoder on the CIFAR-10 dataset. The system learns compressed feature representations (embeddings) of images to clean noisy inputs and retrieve visually similar images efficiently.

## 🚀 Features

*   **Autoencoder Architecture**: A convolutional neural network that compresses images into a lower-dimensional latent space.
*   **Image Denoising**: Capable of removing noise from images as part of the reconstruction process.
*   **Similarity Search**: Retrieves the most similar images from the training set for a given query image using Euclidean distance in the latent space.
*   **Performance Metrics**: Evaluates retrieval performance using Mean Average Precision (mAP).

## 📂 Project Structure

```
CBIR/
├── src/                # Source code modules
│   ├── data_loader.py  # CIFAR-10 loading and preprocessing
│   ├── model.py        # Autoencoder architecture definition
│   ├── train.py        # Training procedure
│   ├── evaluate.py     # Evaluation metrics and scoring
│   └── visualization.py# Image plotting and saving
├── models/             # Directory for saved models
├── data/               # Data files (not tracked by git)
├── output_images/      # Generated results (denoised/retrieved images)
├── main.py             # CLI Entry point
└── requirements.txt    # Python dependencies
```

## 🛠️ Installation

1.  **Clone the repository** (if applicable) or download the source.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

The project is controlled via the `main.py` script.

### 1. Train the Model
Train the autoencoder from scratch.
```bash
python main.py train --epochs 20 --batch_size 128
```
This will save the model to `models/autoencoder.h5`.

### 2. Evaluate Performance
Calculate the Mean Average Precision score.
```bash
python main.py evaluate --n_test_samples 1000
```

### 3. Run Retrieval Demo
Perform a search for a specific test image.
```bash
python main.py demo --index 1 --n_samples 10
```
Results will be saved in `output_images/`.

### 4. Denoise Demo
Visualize the denoising capability of the autoencoder.
```bash
python main.py denoise
```

## 📊 Results

The model learns to map 32x32 RGB images to a robust feature space.
*   **Denoising**: The model can effectively remove Gaussian noise from inputs.
*   **Retrieval**: Queries return semantically similar images based on visual features (color, shape, texture) rather than just raw pixel matching.

## 📝 License

This project is open-source and available for educational purposes.
