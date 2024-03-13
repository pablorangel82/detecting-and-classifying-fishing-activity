import numpy as np

from fusioner import do_fuzzy_inference, discovery_max_pert_per_fishing_spot
from data_loader import load_fishing_spots, get_image_data
from fuzzy import fuzzy_inference
from metrics import metric_mcc
import tensorflow as tf

#values REF: https://dwbi1.wordpress.com/2022/10/05/mcc-formula-for-multiclass-classification/
def mcc_test():
    table = []
    table.append([67, 17, 1])
    table.append([42, 8, 0])
    table.append([67, 20, 0])
    table.append([97, 32, 6])
    table.append([120, 316, 45])
    table.append([127, 356, 46])
    table.append([121, 38, 4])
    table.append([92, 25, 2])
    table.append([81, 27, 1])
    table.append([186, 161, 52])
    print(metric_mcc(table))

def gpu():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def test_fuzzy_matching():
    domain_probability = np.arange(0, 1.1, 0.1)
    domain_proximity = np.arange(0, 11, 1)
    sets = fuzzy_sets(domain_probability, domain_proximity)
    res = fuzzy_inference(1, 0, sets[0], sets[1],sets[2])
    print(res)
    res = fuzzy_inference(0, 15, sets[0], sets[1], sets[2])
    print(res)
    res = fuzzy_inference(0.5, 7, sets[0], sets[1], sets[2])
    print(res)

def test_purse_seine_inference():
    probs=[0.3, 0.35, 0.1, 0.4, 0.35]
    all_areas = load_fishing_spots()
    file_name = '7572518792420.0-9A[fixed_gear,256.7701268266426,2597.3020634019053]'
    type, x, y = get_image_data(file_name)
    probs = do_fuzzy_inference(all_areas, probs, file_name)


    print(probs)

test_purse_seine_inference()