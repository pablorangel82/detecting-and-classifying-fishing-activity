import math

import numpy as np

from fuzzy import Fuzzy
from data_loader import get_image_data


class FusionAgent:

    def __init__(self, cnn, fishing_spots):
        self.cnn = cnn
        self.fishing_spots = fishing_spots

    def get_max_value(self, values):
        max_value = values[0]
        max_index = 0
        for i in range(len(values)):
            if values[i] > max_value:
                max_value = values[i]
                max_index = i
        return max_index, max_value

    def get_closest_area(self, x, y):
        distance = -1
        closest_area = ''
        closest_distance = -1
        closest_type = -1
        for i in range(len(self.fishing_spots)):
            for area in self.fishing_spots[i]:
                x_ref = area[0]
                y_ref = area[1]
                distance = math.sqrt(((x - x_ref) ** 2) + ((y - y_ref) ** 2))
                radius = area[2]
                if distance < radius:
                    if distance < closest_distance or closest_distance == -1:
                        closest_area = area
                        closest_distance = distance
                        if i != 0 and i != 1:
                            closest_type = i + 1
                        else:
                            closest_type = i
        return closest_area, closest_distance, closest_type

    def discovery_max_pert_per_fishing_spot(self, areas, prob, x, y):
        max_pert = -1
        for i in range(len(areas)):
            x_ref = areas[i][0]
            y_ref = areas[i][1]
            radius = areas[i][2]
            distance = math.sqrt(((x - x_ref) ** 2) + ((y - y_ref) ** 2))
            if distance < radius:
                fuzzy = Fuzzy(distance, prob, radius)
                res = fuzzy.execute()
                if res > max_pert or max_pert == -1:
                    max_pert = res
        return max_pert

    def do_fuzzy_inference(self, probs_acquired, file_name):
        type, x, y = get_image_data(file_name)
        results = [0, 0, 0, 0, 0]
        results[0] = probs_acquired[0]
        results[1] = probs_acquired[1]
        results[2] = probs_acquired[2]
        results[3] = probs_acquired[3]
        results[4] = probs_acquired[4]

        closest_area, closest_distance, closest_type = self.get_closest_area(x, y)
        fuzzy = Fuzzy(closest_distance, probs_acquired[closest_type], closest_area[2])
        results[closest_type] = max(probs_acquired[closest_type], fuzzy.execute())

        return results

    #Not used in Fusion Conference paper
    def classify_method2(self, image, file_name):
        input = np.array([image])
        probs = self.cnn.model.predict(input, verbose=0)
        results = probs[0]
        if max(results) != results[2]:
            type, x, y = get_image_data(file_name)
            #drifting_longlines
            results[0]=  self.discovery_max_pert_per_fishing_spot(self.fishing_spots[0], results[0], x, y)
            # fixed_gear
            results[1] = self.discovery_max_pert_per_fishing_spot(self.fishing_spots[1], results[1], x, y)
            #purse_seines
            results[3] = self.discovery_max_pert_per_fishing_spot(self.fishing_spots[2], results[3], x, y)
            #trawlers
            results[4] = self.discovery_max_pert_per_fishing_spot(self.fishing_spots[3], results[4], x, y)
        return self.get_max_value(results)


    def classify(self, image, file_name):
        input = np.array([image])
        probs = self.cnn.model.predict(input, verbose=0)
        results = self.do_fuzzy_inference(probs[0], file_name)
        return self.get_max_value(results)
