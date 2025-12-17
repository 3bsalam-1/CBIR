import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
import os

def retrieve_closest_elements(test_code, test_label, learned_codes, y_train):
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

    # Reshape labels to match the shape of distances
    labels = labels.reshape(distances.shape)

    # Stack distances, labels, and learned_code_index along the last axis
    distance_with_labels = np.stack((distances, labels, learned_code_index), axis=-1)
    sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 0].argsort()]

    sorted_distances = 28 - sorted_distance_with_labels[:, 0]
    sorted_labels = sorted_distance_with_labels[:, 1]
    sorted_indexes = sorted_distance_with_labels[:, 2]
    return sorted_distances, sorted_labels, sorted_indexes

def compute_average_precision_score(test_codes, test_labels, learned_codes, n_samples, y_train):
    out_labels = []
    out_distances = []
    retrieved_elements_indexes = []
    for i in range(len(test_codes)):
        sorted_distances, sorted_labels, sorted_indexes = retrieve_closest_elements(
            test_codes[i], test_labels[i], learned_codes, y_train
        )
        out_distances.append(sorted_distances[:n_samples])
        out_labels.append(sorted_labels[:n_samples])
        retrieved_elements_indexes.append(sorted_indexes[:n_samples])

    out_labels = np.array(out_labels)
    # Ensure directory exists
    os.makedirs('data/computed_results', exist_ok=True)
    
    out_labels_file_name = 'data/computed_results/out_labels_{}.npy'.format(n_samples)
    np.save(out_labels_file_name, out_labels)

    out_distances_file_name = 'data/computed_results/out_distances_{}.npy'.format(n_samples)
    out_distances = np.array(out_distances)
    np.save(out_distances_file_name, out_distances)
    
    score = label_ranking_average_precision_score(out_labels, out_distances)
    return score
