import math

import numpy as np
import tensorflow as tf

from cnn import cnn_training
from data_loader import load_images, get_image_data, load_fishing_spots
from fuzzy import fuzzy_inference
from metrics import show_metrics

train_dir = 'images/train'
val_dir = 'images/val'
test_dir = 'images/test'


def get_max_value(values):
    max_value = values[0]
    max_index = 0
    for i in range(len(values)):
        if values[i] > max_value:
            max_value = values[i]
            max_index = i
    return max_index, max_value

def get_closest_area(all_areas, x, y):
    distance = -1
    closest_area = ''
    closest_distance = -1
    closest_type = -1
    for i in range(len(all_areas)):
        for area in all_areas[i]:
            x_ref = area[0]
            y_ref = area[1]
            distance = math.sqrt(((x - x_ref) ** 2) + ((y - y_ref) ** 2))
            radius = area[2]
            if distance < radius:
                if distance < closest_distance or closest_distance == -1:
                    closest_area = area
                    closest_distance = distance
                    if i != 0 and i != 1:
                        closest_type = i+1
                    else:
                        closest_type = i
    return closest_area, closest_distance, closest_type

def discovery_max_pert_per_fishing_spot(areas, prob, x, y):
    max_pert = -1
    for i in range(len(areas)):
        x_ref = areas[i][0]
        y_ref = areas[i][1]
        radius = areas[i][2]
        distance = math.sqrt(((x - x_ref) ** 2) + ((y - y_ref) ** 2))
        if distance < radius:
            res = fuzzy_inference(distance, prob, radius)
            if res > max_pert or max_pert == -1:
                max_pert = res
    return max_pert

def do_fuzzy_inference(set_areas, probs_acquired, file_name):
    type, x, y = get_image_data(file_name)
    results = [0,0,0,0,0]
    results[0] = probs_acquired[0]
    results[1] = probs_acquired[1]
    results[2] = probs_acquired[2]
    results[3] = probs_acquired[3]
    results[4] = probs_acquired[4]

    closest_area, closest_distance, closest_type = get_closest_area(set_areas, x, y)
    results[closest_type] = max(probs_acquired[closest_type],fuzzy_inference(closest_distance, probs_acquired[closest_type], closest_area[2]))

    return results

# def do_fuzzy_inference(set_areas, probs_acquired, file_name):
#     type, x, y = get_image_data(file_name)
#     results = [0,0,0,0,0]
#
#     if max(probs_acquired) != probs_acquired[2]:
#         #drifting_longlines
#         results[0]=  discovery_max_pert_per_fishing_spot(set_areas[0], probs_acquired[0], x, y)
#         # fixed_gear
#         results[1] = discovery_max_pert_per_fishing_spot(set_areas[1], probs_acquired[1], x, y)
#         #not_fishing
#         results[2] = probs_acquired[2]
#         #purse_seines
#         results[3] = discovery_max_pert_per_fishing_spot(set_areas[2], probs_acquired[3], x, y)
#         #trawlers
#         results[4] = discovery_max_pert_per_fishing_spot(set_areas[3], probs_acquired[4], x, y)
#         return results
#     return probs_acquired
def run():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train_dataset, val_dataset, test_dataset = load_images(train_dir, val_dir, test_dir)
    all_areas = load_fishing_spots()
    model = cnn_training(train_dataset, val_dataset, 5)

    test_file_names = test_dataset.filenames

    matrix_with_fuzzy = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],[0, 0, 0, 0, 0]])
    table_with_fuzzy = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    matrix_without_fuzzy = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    table_without_fuzzy = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

    total = 0

    for images, labels in test_dataset:
        probs = model.predict(images)
        count = len(images)
        for i in range(count):

            results_with_fuzzy = do_fuzzy_inference(all_areas,probs[i],test_file_names[i])
            max_index_pert, max_value_pert = get_max_value(results_with_fuzzy)
            max_index_label, max_value_label = get_max_value(labels[i])
            table_with_fuzzy[max_index_label][0] += 1
            table_with_fuzzy[max_index_pert][1] += 1
            if max_index_pert == max_index_label:
                table_with_fuzzy[max_index_pert][2] += 1
            matrix_with_fuzzy[max_index_label][max_index_pert] = matrix_with_fuzzy[max_index_label][max_index_pert] + 1

            max_index_prob, max_value_prob = get_max_value(probs[i])
            table_without_fuzzy[max_index_label][0] += 1
            table_without_fuzzy[max_index_prob][1] += 1
            if max_index_prob == max_index_label:
                table_without_fuzzy[max_index_prob][2] += 1
            matrix_without_fuzzy[max_index_label][max_index_prob] = matrix_without_fuzzy[max_index_label][max_index_prob] + 1


        total += count
        print(total)
        if total >= test_dataset.samples:
            print('Results without Fuzzy')
            show_metrics(matrix_without_fuzzy, table_without_fuzzy)
            print('Results using Fuzzy')
            show_metrics(matrix_with_fuzzy, table_with_fuzzy)
            exit(0)
