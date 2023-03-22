import pandas as pd
# import numpy as np
# import copy
import itertools
import time
import matplotlib.pyplot as plt
import random

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

        if type=='random':
            # connect random nodes
            possible_arcs = [arc for arc in list(itertools.combinations(all_nodes,2))]
            nr_connections = random.randint(int(0.2*len(possible_arcs)),int(0.8*len(possible_arcs)))
            arcs = random.choices(possible_arcs, k=nr_connections, replace=False)
            print(f"{nr_connections}/{len(possible_arcs)} random arcs")

            self.true_dbn.addArcs(arcs)


        if type=='close':
            # connect nodes randomly based on difference in timesteps
            arcs = []
            possible_arcs = [arc for arc in list(itertools.combinations(all_nodes,2))]
            for possible_arc in possible_arcs:
                rand = random.random()
                if rand < 1/(6 + (possible_arc[1]//timesteps-possible_arc[0]//timesteps)**2):
                    # print(rand, 1/(6 + (possible_arc[1]//timesteps-possible_arc[0]//timesteps)**2))
                    arcs.append(possible_arc)

            self.true_dbn.addArcs(arcs)

        return 


    def generate_CPTs(self):
        self.true_dbn.generateCPTs()
        return


    def generate_data(self,train_items=10000, test_items=1000):
        self.train_data,_=gum.generateSample(self.true_dbn,train_items,None,False)
        self.train_data = self.train_data.reindex(sorted(self.train_data.columns, key=lambda x: x[::-1]), axis=1)
        
        self.test_data,_=gum.generateSample(self.true_dbn,test_items,None,False)
        self.test_data = self.test_data.reindex(sorted(self.test_data.columns, key=lambda x: x[::-1]), axis=1)
        return


    def learn_dbn(self,timesteps=2, structure='hill', parameter='test'):
        discretizer = skbn.BNDiscretizer(defaultDiscretizationMethod='uniform',defaultNumberOfBins=5,discretizationThreshold=25)
        # Create nodes
        template = gum.BayesNet()
        for name in self.train_data:
            template.add(discretizer.createVariable(name, self.train_data[name]))

        # Set up learner
        learner = gum.BNLearner(self.train_data, template)
        # TODO: More structure and parameter algorithms
        if structure == 'hill':
            learner.useGreedyHillClimbing()
        if structure == 'tabu':
            learner.useLocalSearchWithTabuList()
        if structure == 'k2':
            learner.useK2([i for i in range(len(self.train_data.columns))])
        if parameter == 'a':
            print("parameter algorithm")

        # Force no back in time arcs
        for name1 in self.train_data:
            for name2 in self.train_data:
                if int(name1[-1]) > int(name2[-1]):
                    learner.addForbiddenArc(name1,name2)
                if int(name1[-1]) + timesteps <= int(name2[-1]):
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
        xTest, yTest = self.test_data.loc[:, self.test_data.columns != self.target], self.test_data[self.target]
        return self.ClassfromBN.score(xTest, yTest.apply(lambda x: x==1))

    
    def time_test(self, timesteps=2, cpt_repetitions=5, targets=["c2"], train_items=10000, test_items=1000, structure='hill', parameter='test'):
        all_scores = pd.DataFrame()#columns=targets)
        for i in range(cpt_repetitions):
            self.generate_CPTs()

            # TODO: WAY TO LOOK AT DIFFERENT CPTs

            # for j in range(data_repetitions):
            # print(f"Cpt {i + 1} and Data {j + 1}")
            self.generate_data(train_items=train_items, test_items=test_items)
            bn = self.learn_dbn(timesteps=timesteps, structure=structure, parameter=parameter)

            scores = {}
            for target in targets:
                self.classifier_from_dbn(bn, target)
                scores[f'{target}.{timesteps}'] = [self.score_classifier()]
            all_scores = pd.concat([all_scores, pd.DataFrame(scores, index=[f't={timesteps}'], )], ignore_index=True)
            # return all_scores, bn
        return all_scores

if __name__ == '__main__':
    print("begin")
    start_time = time.time()

    timesteps = 3
    # nodes = 4
    bayestest = Bayes_Test(timesteps=timesteps) 
    print("Created")

    targets = list(sorted(bayestest.true_dbn.names(), key=lambda x: x[::-1]))[-3:]

    fig, axs = plt.subplots(1, timesteps)

    for timesteps in range(1, timesteps + 1):
        scores = bayestest.time_test(targets=targets, timesteps=timesteps)

        print("--- %s seconds ---" % (time.time() - start_time))

        axs[timesteps-1].boxplot(scores)

    plt.show()