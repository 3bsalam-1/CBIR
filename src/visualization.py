import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import label_ranking_average_precision_score
import os

def retrieve_closest_images(test_element, test_label, encoder, x_train, y_train, n_samples=10, save_dir='output_images'):
    learned_codes = encoder.predict(x_train)
    learned_codes = learned_codes.reshape(learned_codes.shape[0],
                                          learned_codes.shape[1] * learned_codes.shape[2] * learned_codes.shape[3])

    test_code = encoder.predict(np.array([test_element]))
    test_code = test_code.reshape(test_code.shape[1] * test_code.shape[2] * test_code.shape[3])

    distances = []

    for code in learned_codes:
        distance = np.linalg.norm(code - test_code)
        distances.append(distance)
    nb_elements = learned_codes.shape[0]
    distances = np.array(distances)
    learned_code_index = np.arange(nb_elements)
    labels = np.copy(y_train).astype('float32')
    labels[labels != test_label] = -1
    labels[labels == test_label] = 1
    labels[labels == -1] = 0

    # Reshape labels and learned_code_index to match the shape of distances
    labels = labels.reshape(distances.shape)
    learned_code_index = learned_code_index.reshape(distances.shape)

    distance_with_labels = np.stack((distances, labels, learned_code_index), axis=-1)
    sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 0].argsort()]

    sorted_distances = 28 - sorted_distance_with_labels[:, 0]
    sorted_labels = sorted_distance_with_labels[:, 1]
    sorted_indexes = sorted_distance_with_labels[:, 2]
    kept_indexes = sorted_indexes[:n_samples]

    score = label_ranking_average_precision_score(np.array([sorted_labels[:n_samples]]), np.array([sorted_distances[:n_samples]]))

    print("Average precision ranking score for tested element is {}".format(score))

    original_image = test_element
    original_image = cv2.resize(original_image, (140, 140))
    fig, axs = plt.subplots(2, n_samples, figsize=(20, 10))
    
    # Try using tight layout or just disable backend interactive mode if running headless
    # fig_manager = plt.get_current_fig_manager()
    # fig_manager.window.state('zoomed') # This might fail on non-windows or non-interactive
    
    axs[0, 5].imshow(original_image)
    axs[0, 5].set_title('Original Image')
    
    os.makedirs(save_dir, exist_ok=True)

    retrieved_images = x_train[int(kept_indexes[0]), :]
    for i in range(1, n_samples):
        retrieved_images = np.hstack((retrieved_images, x_train[int(kept_indexes[i]), :]))
    for i in range(n_samples):
        retrieved_images = x_train[int(kept_indexes[i]), :]
        retrieved_images = cv2.resize(retrieved_images, (140, 140))
        axs[1, i].imshow(retrieved_images)
        axs[1, i].axis('off')
        axs[0, i].axis('off')
        # Use RGB for saving to ensure correct colors if not converting to BGR for cv2
        # cv2 uses BGR, but we loaded generic cifar10 which is RGB usually.
        # But here assuming it's standard 0-1 float RGB. cv2 expects 0-255.
        
        cv2.imwrite(f'{save_dir}/retrieved_results_{i+1}.jpg', 255 * cv2.cvtColor(retrieved_images, cv2.COLOR_RGB2BGR))
        
    axs[1, 5].set_title('Retrieved Images')
    plt.tight_layout()
    # plt.show() # Blocking call, might want to avoid or safeguard

    cv2.imwrite(f'{save_dir}/original_image.jpg', 255 * cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    print(f"Images saved to {save_dir}")

def plot_denoised_images(autoencoder, x_test_noisy, save_dir='output_images'):
    denoised_images = autoencoder.predict(x_test_noisy)
    test_img = x_test_noisy[1]
    resized_test_img = cv2.resize(test_img, (280, 280))
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(resized_test_img)
    plt.title('Input Image')
    plt.axis('off')
    
    output = denoised_images[1]
    resized_output = cv2.resize(output, (280, 280))
    
    plt.subplot(122)
    plt.imshow(resized_output)
    plt.title('Denoised Image')
    plt.axis('off')
    
    os.makedirs(save_dir, exist_ok=True)
    # plt.show() # Blocking
    
    cv2.imwrite(f'{save_dir}/noisy_image.jpg', 255 * cv2.cvtColor(resized_test_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{save_dir}/denoised_image.jpg', 255 * cv2.cvtColor(resized_output, cv2.COLOR_RGB2BGR))
    print(f"Denoised images saved to {save_dir}")
