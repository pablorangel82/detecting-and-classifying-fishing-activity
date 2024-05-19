from cnn import CNN
import numpy as np
import tensorflow as tf
from fusion_agent import FusionAgent
from metrics import show_metrics
from data_loader import load_images, load_fishing_spots

train_dir = 'images/train'
val_dir = 'images/val'
test_dir = 'images/test'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def load_and_test():
    print('Loading and Testing')
    train_dataset, val_dataset, test_dataset = load_images(train_dir, val_dir, test_dir)
    cnn = CNN(model_path='output/model.keras')
    test(cnn, test_dataset)


def fusion_results():
    print('Fusion Results')
    train_dataset, val_dataset, test_dataset = load_images(train_dir, val_dir, test_dir)
    cnn = CNN(model_path='fusion_results/model.keras')
    test(cnn, test_dataset)


def train_and_test():
    print('Training and Testing')
    train_dataset, val_dataset, test_dataset = load_images(train_dir, val_dir, test_dir)
    cnn = CNN(iterations=10,train_dataset=train_dataset, val_dataset=val_dataset, number_of_classes=5)
    cnn.train()
    test(cnn, test_dataset)
    answer = input('Do you want to save this model? Yes(y) or No(n).  ')
    if answer.lower() == 'y':
        print('Saving the model...')
        cnn.save()
        print('Saved.')


def test(cnn, test_dataset):
    fishing_spots = load_fishing_spots()
    test_file_names = test_dataset.filenames
    matrix_with_fuzzy = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    table_with_fuzzy = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

    total = 0

    fusion_agent = FusionAgent(cnn, fishing_spots)
    for images, labels in test_dataset:
        count = len(images)
        for i in range(count):
            max_index_pert, max_value_pert = fusion_agent.classify(images[i], test_file_names[i])
            max_index_label, max_value_label = fusion_agent.get_max_value(labels[i])
            table_with_fuzzy[max_index_label][0] += 1
            table_with_fuzzy[max_index_pert][1] += 1
            if max_index_pert == max_index_label:
                table_with_fuzzy[max_index_pert][2] += 1
            matrix_with_fuzzy[max_index_label][max_index_pert] = matrix_with_fuzzy[max_index_label][max_index_pert] + 1

        total += count
        print('Predicted so far: ' + str(total))
        if total >= test_dataset.samples:
            print('Results:')
            show_metrics(matrix_with_fuzzy, table_with_fuzzy)
            return
