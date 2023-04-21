import pandas as pd
import re
from syntheticDBN2 import Generator
from pgmpy.estimators import PC, MmhcEstimator, BDeuScore, HillClimbSearch
import pyAgrum as gum
import pyAgrum.lib.dynamicBN as gdyn
import pyAgrum.skbn as skbn

class Node:
    def __init__(self, name, timestep, values=2):
        self.name = name
        self.timestep = timestep
        self.values = values

    def __str__(self):
        return f"{self.name}{self.timestep}-{self.values}"
    def __repr__(self):
        return f"{self.name}{self.timestep}-{self.values}"

class DBN():
    def __init__(self, data, values=None):
        self.data = data
        self.nodes = []
        self.timesteps = 0
        for column in self.data.columns:
            name, timestep, _ = re.split('(\d+)', column) 
            if not values:
                values = max(self.data[column]) - min(self.data[column]) + 1
            self.nodes.append(Node(name, timestep, values=values))

            if int(timestep) > self.timesteps:
                self.timesteps = int(timestep)
        self.nr_nodes = int(len(self.nodes)/self.timesteps)
        # print(self.nodes)
    
    def learn_causal_structure(self, timestep):
        # TODO: FIND BEST STRUCTURE LEARNING METHOD THAT HAS A WAY OF GETTING CERTAINTY

        train = data[[col for col in data.columns if int(col[-1]) == timestep]]
        train.columns = [col[:-1] for col in train.columns]
        # print(train.columns)

        #### PGMPY ####

        # THIS IS A CONSTRAINED BASED METHOD WHICH IS PROVEN TO BE WORSE?
        # est = PC(train)
        # skel, seperating_sets = est.build_skeleton(significance_level=0.01)
        # print("Undirected edges: ", skel.edges())
        # pdag = est.skeleton_to_pdag(skel, seperating_sets)
        # print("PDAG edges:       ", pdag.edges())

        mmhc = MmhcEstimator(train)
        skeleton = mmhc.mmpc()
        print("Part 1) Skeleton: ", skeleton.edges())

        # use hill climb search to orient the edges:
        hc = HillClimbSearch(train)
        model = hc.estimate(tabu_length=10, white_list=skeleton.to_directed().edges(), scoring_method=BDeuScore(train), show_progress=False)
        print("Part 2) Model:    ", model.edges())

        #### PYAGRUM ####

        # discretizer = skbn.BNDiscretizer(defaultDiscretizationMethod='uniform',defaultNumberOfBins=5,discretizationThreshold=25)

        # # Create nodes
        # template = gum.BayesNet()
        # for name in train:
        #     template.add(discretizer.createVariable(name, train[name]))

        # learner = gum.BNLearner(train, template)
        # bn = learner.learnDAG()
        # print(bn.arcs())

        return



if __name__ == '__main__':
    generator = Generator(timesteps=4, nodes=5, structure='serie', values=4) 
    data = generator.generate_data(train_items=10000)
    dbn = DBN(data)
    for t in range(4):
        dbn.learn_causal_structure(t)
    
