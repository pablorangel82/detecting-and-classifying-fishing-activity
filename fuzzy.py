import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def fuzzy_inference(distance_value, probability_value, radius):
    domain_probability = np.arange(0, 1.1, 0.1)
    domain_distance = np.arange(0, radius+1, 1)

    distance = ctrl.Antecedent(domain_distance, 'distance')
    probability = ctrl.Antecedent(domain_probability, 'probability')
    output = ctrl.Consequent(np.arange(0, 1.1, .1), 'output')

    distance['close'] = fuzz.trapmf(distance.universe, [0, 0, radius * 0.05, radius * 0.1])
    distance['medium'] = fuzz.trimf(distance.universe, [radius * 0.05, radius * 0.35, radius * 0.5])
    distance['far'] = fuzz.trapmf(distance.universe, [radius * 0.425, radius * 0.8, radius, radius])

    probability['low'] = fuzz.trapmf(probability.universe, [0, 0, 0.1, 0.4])
    probability['average'] = fuzz.trimf(probability.universe, [0.3, 0.6, 0.9])
    probability['high'] = fuzz.trapmf(probability.universe, [0.8, 0.9, 1, 1])
    # probability.view()

    output['low'] = fuzz.trapmf(output.universe, [0, 0, 0.1, 0.4])
    output['average'] = fuzz.trimf(output.universe, [0.3, 0.6, 0.9])
    output['high'] = fuzz.trapmf(output.universe, [0.8, 0.9, 1, 1])

    rule1 = ctrl.Rule(distance['close'] & probability['low'], output['average'])
    rule2 = ctrl.Rule(distance['close'] & probability['average'], output['high'])
    rule3 = ctrl.Rule(distance['close'] & probability['high'], output['high'])
    rule4 = ctrl.Rule(distance['medium'] & probability['low'], output['low'])
    rule5 = ctrl.Rule(distance['medium'] & probability['average'], output['average'])
    rule6 = ctrl.Rule(distance['medium'] & probability['high'], output['high'])
    rule7 = ctrl.Rule(distance['far'] & probability['low'], output['low'])
    rule8 = ctrl.Rule(distance['far'] & probability['average'], output['low'])
    rule9 = ctrl.Rule(distance['far'] & probability['high'], output['average'])

    rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]

    classifier_ctrl = ctrl.ControlSystem(rules)

    engine = ctrl.ControlSystemSimulation(classifier_ctrl)
    engine.input['distance'] = distance_value
    engine.input['probability'] = probability_value
    engine.compute()

   #output.view()
    res = engine.output['output']

    return res