import pandas as pd
import numpy as np
# import copy
import itertools
import time
import matplotlib.pyplot as plt
import random

import pyAgrum as gum
import pyAgrum.lib.dynamicBN as gdyn
import pyAgrum.skbn as skbn
# import pyAgrum.lib.bn_vs_bn as bnvsbn

class Generator:
    def __init__(self, nodes=4, timesteps=3, structure='serie', values=2):
        self.nodes = nodes
        self.timesteps = timesteps
        
        self.generate_DBN(structure=structure, values=values)

        return 

    def generate_DBN(self, structure='serie', values=2):
        # Initialize true_dbn
        twodbn = gum.BayesNet()
        # Create nodes
        all_nodes = []
        all_names = []

        names = [f"{chr(ord('`')+(x%26 + 1))}"*(np.ceil((x+1)/26).astype(int)) for x in range(self.nodes)]
        # print(names)
        for t in ['0','t']:
            for i in range(self.nodes):
                name = names[i]+t
                all_names.append(name)
                all_nodes.append(twodbn.add(gum.LabelizedVariable(name,name,values)))

        if structure=='serie':
            #OC Stands for one causal structure!!!

            arcs = [arc for arc in list(itertools.combinations(all_nodes,2)) if arc[1] % self.nodes != 0 and arc[1] - arc[0] < 2]

            for node in range(self.nodes):
                arcs.append((node, node+self.nodes))

            twodbn.addArcs(arcs)

            twodbn.generateCPTs()

            self.true_dbn=gdyn.unroll2TBN(twodbn,self.timesteps)

        return
    
    def generate_data(self,train_items=10000, test_items=1000):
        self.train_items = train_items
        self.train_data,_=gum.generateSample(self.true_dbn,train_items,None,False)
        self.train_data = self.train_data.reindex(sorted(self.train_data.columns, key=lambda x: (len(x), x[::-1], x[-1])), axis=1)
        self.train_data = self.train_data.astype('int')
        
        self.test_items = test_items
        self.test_data,_=gum.generateSample(self.true_dbn,test_items,None,False)
        self.test_data = self.test_data.reindex(sorted(self.test_data.columns, key=lambda x: x[::-1]), axis=1)
        return self.train_data


class Bayes_Test:
    def __init__(self, data, test_data=None):
        self.data = data
        

        return 


    def learn_dbn(self,timesteps=2, structure='hill', parameter='test'):
        # discretizer = skbn.BNDiscretizer(defaultDiscretizationMethod='uniform',defaultNumberOfBins=5,discretizationThreshold=25)
        # # Create nodes
        # template = gum.BayesNet()
        # for name in self.train_data:
        #     template.add(discretizer.createVariable(name, self.train_data[name]))

        # # Set up learner
        # learner = gum.BNLearner(self.train_data, template)
        # # TODO: More structure and parameter algorithms
        # if structure == 'hill':
        #     learner.useGreedyHillClimbing()
        # if structure == 'tabu':
        #     learner.useLocalSearchWithTabuList()
        # if structure == 'k2':
        #     learner.useK2([i for i in range(len(self.train_data.columns))])
        # if parameter == 'a':
        #     print("parameter algorithm")

        # # Force no back in time arcs
        # for name1 in self.train_data:
        #     for name2 in self.train_data:
        #         if int(name1[-1]) > int(name2[-1]):
        #             learner.addForbiddenArc(name1,name2)
        #         if int(name1[-1]) + timesteps <= int(name2[-1]):
        #             learner.addForbiddenArc(name1,name2)

        # bn = learner.learnBN()
        return bn
    
    def learn_causal_structure(self, structure='hill', parameter='test', t=2, fold=0):
        # Train on first timestep
        # train = self.train_data.iloc[:,0:self.nodes]
        fold_items = int(np.floor(self.train_items/self.folds))
        train = self.train_data.iloc[fold_items*fold:fold_items+fold_items*fold,self.nodes*t:self.nodes+self.nodes*t]
        train.columns = [col[0] for col in train.columns]

        discretizer = skbn.BNDiscretizer(defaultDiscretizationMethod='uniform',defaultNumberOfBins=5,discretizationThreshold=25)

        # Create nodes
        template = gum.BayesNet()
        for name in train:
            template.add(discretizer.createVariable(name, train[name]))

        # Set up learner
        learner = gum.BNLearner(train, template)
        # TODO: More structure and parameter algorithms
        # TODO: Use multiple algorithms
        if structure == 'hill':
            learner.useGreedyHillClimbing()
        if structure == 'tabu':
            learner.useLocalSearchWithTabuList()
        if structure == 'k2':
            learner.useK2([i for i in range(len(self.train_data.columns))])
        if parameter == 'a':
            print("parameter algorithm")
        learner.setEpsilon(1e-10)

        # TODO: Learn DAG or BN? 
        # dag = learner.learnDAG()
        # return dag
        bn = learner.learnBN()

        # bns = [bn]
        # for t in range(1, self.timesteps):
        #     train = self.train_data.iloc[:,self.nodes*t:self.nodes+self.nodes*t]
        #     train.columns = [col[0] for col in train.columns]
        #     # Set up learner
        #     learner = gum.BNLearner(train, template)
        #     # TODO: More structure and parameter algorithms
        #     # TODO: Use multiple algorithms
        #     if structure == 'hill':
        #         learner.useGreedyHillClimbing()
        #     if structure == 'tabu':
        #         learner.useLocalSearchWithTabuList()
        #     if structure == 'k2':
        #         learner.useK2([i for i in range(len(self.train_data.columns))])
        #     if parameter == 'a':
        #         print("parameter algorithm")
        #     learner.setEpsilon(1e-10)
        #     learner.setInitialDAG(bn.dag())
        #     bn = learner.learnBN()
        #     bns.append(bn)
        # return bns
        return bn
    

    def learn_time_structure(self, structure='hill', t=2, parameter='test'):
        subtimes = []
        for timestep in range(self.timesteps):
            if timestep + t <= self.timesteps:
                subtimes.append([i for i in range(timestep, timestep + t)])

        bns = []
        for subtime in subtimes:
            # Train on first timestep
            train = self.train_data[[column for column in self.train_data.columns if int(column[-1]) in subtime]]
            
            discretizer = skbn.BNDiscretizer(defaultDiscretizationMethod='uniform',defaultNumberOfBins=5,discretizationThreshold=25)

            # Create nodes
            template = gum.BayesNet()
            for name in train:
                template.add(discretizer.createVariable(name, train[name]))

            # Set up learner
            learner = gum.BNLearner(train, template)
            # TODO: More structure and parameter algorithms
            # TODO: Use multiple algorithms
            if structure == 'hill':
                learner.useGreedyHillClimbing()
            if structure == 'tabu':
                learner.useLocalSearchWithTabuList()
            if structure == 'k2':
                learner.useK2([i for i in range(len(train.columns))])
            if parameter == 'a':
                print("parameter algorithm")
            learner.setEpsilon(1e-10)

            # Force no back in time and in same timestep arcs
            for name1 in train:
                for name2 in train:
                    if int(name1[-1]) > int(name2[-1]):
                        learner.addForbiddenArc(name1,name2)
                    # if int(name1[-1]) + t < int(name2[-1]):
                    #     learner.addForbiddenArc(name1,name2)
                    if int(name1[-1]) == int(name2[-1]):
                        learner.addForbiddenArc(name1,name2)

            # TODO: Learn DAG or BN? 
            # dag = learner.learnDAG()
            # return dag
            bns.append(learner.learnBN())
        return bns


    def classifier_from_dbn(self,bn, target):
        # self.target = target
        # self.ClassfromBN = skbn.BNClassifier(significant_digit = 7)
        # self.ClassfromBN.fromTrainedModel(bn = bn, targetAttribute = self.target)
        return


    def score_classifier(self):
        # self.ClassfromBN.fromTrainedModel(self.true_dbn,targetAttribute=self.target)
        # xTest, yTest = self.test_data.loc[:, self.test_data.columns != self.target], self.test_data[self.target]
        return self.ClassfromBN.score(xTest, yTest.apply(lambda x: x==1))

    
    def time_test(self, timesteps=2, cpt_repetitions=5, targets=["c2"], train_items=10000, test_items=1000, structure='hill', parameter='test'):
        # all_scores = pd.DataFrame()#columns=targets)
        # for i in range(cpt_repetitions):
        #     self.generate_CPTs()

        #     # TODO: WAY TO LOOK AT DIFFERENT CPTs

        #     # for j in range(data_repetitions):
        #     # print(f"Cpt {i + 1} and Data {j + 1}")
        #     self.generate_data(train_items=train_items, test_items=test_items)
        #     bn = self.learn_dbn(timesteps=timesteps, structure=structure, parameter=parameter)

        #     scores = {}
        #     for target in targets:
        #         self.classifier_from_dbn(bn, target)
        #         scores[f'{target}.{timesteps}'] = [self.score_classifier()]
        #     all_scores = pd.concat([all_scores, pd.DataFrame(scores, index=[f't={timesteps}'], )], ignore_index=True)
        #     # return all_scores, bn
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