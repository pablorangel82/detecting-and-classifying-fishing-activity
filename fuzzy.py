import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class Fuzzy:

    def __init__(self, distance_value, probability_value, radius):
        self.domain_probability = np.arange(0, 1.1, 0.1)
        self.domain_distance = np.arange(0, radius + 1, 1)
        self.radius = radius
        proximity = ctrl.Antecedent(self.domain_distance, 'proximity')
        probability = ctrl.Antecedent(self.domain_probability, 'probability')
        output = ctrl.Consequent(np.arange(0, 1.1, .1), 'fusion_results')

        proximity['close'] = fuzz.trapmf(proximity.universe, [0, 0, self.radius * 0.05, self.radius * 0.1])
        proximity['medium'] = fuzz.trimf(proximity.universe,
                                         [self.radius * 0.05, self.radius * 0.275, self.radius * 0.5])
        proximity['far'] = fuzz.trapmf(proximity.universe,
                                       [self.radius * 0.425, self.radius * 0.6, self.radius, self.radius])

        probability['low'] = fuzz.trapmf(probability.universe, [0, 0, 0.1, 0.4])
        probability['average'] = fuzz.trimf(probability.universe, [0.3, 0.6, 0.9])
        probability['high'] = fuzz.trapmf(probability.universe, [0.8, 0.9, 1, 1])
        # probability.view()

        output['low'] = fuzz.trapmf(output.universe, [0, 0, 0.1, 0.4])
        output['average'] = fuzz.trimf(output.universe, [0.3, 0.6, 0.9])
        output['high'] = fuzz.trapmf(output.universe, [0.8, 0.9, 1, 1])

        rule1 = ctrl.Rule(proximity['close'] & probability['low'], output['average'])
        rule2 = ctrl.Rule(proximity['close'] & probability['average'], output['high'])
        rule3 = ctrl.Rule(proximity['close'] & probability['high'], output['high'])
        rule4 = ctrl.Rule(proximity['medium'] & probability['low'], output['low'])
        rule5 = ctrl.Rule(proximity['medium'] & probability['average'], output['average'])
        rule6 = ctrl.Rule(proximity['medium'] & probability['high'], output['high'])
        rule7 = ctrl.Rule(proximity['far'] & probability['low'], output['low'])
        rule8 = ctrl.Rule(proximity['far'] & probability['average'], output['low'])
        rule9 = ctrl.Rule(proximity['far'] & probability['high'], output['average'])
        rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]

        classifier_ctrl = ctrl.ControlSystem(rules)
        self.engine = ctrl.ControlSystemSimulation(classifier_ctrl)
        self.engine.input['proximity'] = distance_value
        self.engine.input['probability'] = probability_value

    def execute(self):
        self.engine.compute()
        result = self.engine.output['fusion_results']
        return result
