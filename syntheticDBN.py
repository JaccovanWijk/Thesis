import pandas as pd
# import numpy as np
# import copy
import itertools

import pyAgrum as gum
# import pyAgrum.lib.dynamicBN as gdyn
import pyAgrum.skbn as skbn
# import pyAgrum.lib.bn_vs_bn as bnvsbn

class Bayes_Test:
    def __init__(self, nodes=4, timesteps=3, type='connected'):
        # Initialize true_dbn
        self.true_dbn = gum.BayesNet()

        # Create nodes
        all_nodes = []
        all_names = []
        # TODO: if nodes > 26
        for t in range(timesteps):
            for i in range(nodes):
                name = f"{chr(ord('`')+i+1)}{t}"
                all_names.append(name)
                all_nodes.append(self.true_dbn.add(gum.LabelizedVariable(name,name,2)))

        # Connect nodes based on the type
        if type=='connected':
            # connect same nodes over time and different nodes within timestep
            arcs = [arc for arc in list(itertools.combinations(all_nodes,2)) if (all_names[arc[0]][:-1] == all_names[arc[1]][:-1]) or (all_names[arc[0]][-1] == all_names[arc[1]][-1])]

            self.true_dbn.addArcs(arcs)
        return 


    def generate_CPTs(self):
        self.true_dbn.generateCPTs()
        return


    def generate_data(self,items=10000):
        self.data,_=gum.generateSample(self.true_dbn,items,None,False)
        self.data = self.data.reindex(sorted(self.data.columns, key=lambda x: x[::-1]), axis=1)
        return


    def learn_dbn(self,structure='hill', parameter='test'):
        discretizer = skbn.BNDiscretizer(defaultDiscretizationMethod='uniform',defaultNumberOfBins=5,discretizationThreshold=25)
        # Create nodes
        template = gum.BayesNet()
        for name in self.data:
            template.add(discretizer.createVariable(name, self.data[name]))

        # Set up learner
        learner = gum.BNLearner(self.data, template)
        # TODO: More structure and parameter algorithms
        if structure == 'hill':
            learner.useGreedyHillClimbing()
        if parameter == 'a':
            print("parameter algorithm")

        # Force no back in time arcs
        for name1 in self.data:
            for name2 in self.data:
                if int(name1[-1]) > int(name2[-1]):
                    learner.addForbiddenArc(name1,name2)

        bn = learner.learnBN()
        return bn


    def classifier_from_dbn(self,bn, target):
        self.target = target
        self.ClassfromBN = skbn.BNClassifier(significant_digit = 7)
        self.ClassfromBN.fromTrainedModel(bn = bn, targetAttribute = self.target)
        return


    def score_classifier(self):
        self.ClassfromBN.fromTrainedModel(self.true_dbn,targetAttribute=self.target)
        self.generate_data()
        xTest, yTest = self.data.loc[:, self.data.columns != self.target], self.data[self.target]
        return self.ClassfromBN.score(xTest, yTest.apply(lambda x: x==1))

    
    def test_bayes(self, cpt_repetitions=5, data_repetitions=10, targets=["c2"]):
        all_scores = pd.DataFrame(columns=targets)
        for _ in range(cpt_repetitions):
            self.generate_CPTs()

            # TODO: WAY TO LOOK AT DIFFERENT CPTs

            for _ in range(data_repetitions):
                self.generate_data()
                bn = self.learn_dbn()

                scores = {}
                for target in targets:
                    bayestest.classifier_from_dbn(bn, target)
                    scores[target] = [bayestest.score_classifier()]
                all_scores = pd.concat([all_scores, pd.DataFrame(scores)], ignore_index=True)

        return all_scores



bayestest = Bayes_Test()
all_targets = list(sorted(bayestest.true_dbn.names(), key=lambda x: x[::-1]))

print(bayestest.test_bayes(targets=all_targets))

