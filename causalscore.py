import pandas as pd
import numpy as np
import itertools
import pyAgrum as gum
import pyAgrum.lib.dynamicBN as gdyn
import pyAgrum.skbn as skbn
import pyAgrum.lib.bn_vs_bn as gcm
import random
import multiprocessing as mp
import pyAgrum.lib.bn_vs_bn as bnvsbn

class Generator:
    def __init__(self, items=1000, nodes=4, timesteps=3, structure='serie', values=2, algorithm="HC", score="BDEU"):
        self.items = items
        self.nodes = nodes
        self.timesteps = timesteps
        self.structure = structure
        self.values = values
        self.algorithm = algorithm
        self.scorefunction = score
        self.parameters = {"nodes":self.nodes, "timesteps":self.timesteps, "values":self.values, "structure":self.structure, "algorithm":self.algorithm, "score":self.scorefunction}
        return 

    def generate_DBN(self):
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
                all_nodes.append(twodbn.add(gum.LabelizedVariable(name,name,self.values)))

        if self.structure=='serie':
            #OC Stands for one causal structure!!!
            arcs = [arc for arc in list(itertools.combinations(all_nodes,2)) if arc[1] % self.nodes != 0 and arc[1] - arc[0] < 2]

            for node in range(self.nodes):
                arcs.append((node, node+self.nodes))

            twodbn.addArcs(arcs)
            twodbn.generateCPTs()
            self.true_dbn=gdyn.unroll2TBN(twodbn,self.timesteps)

        if self.structure=='islands':
            #OC Stands for one causal structure!!!
            arcs = [arc for arc in list(itertools.combinations(all_nodes,2)) if arc[1] % int(self.nodes/2) != 0 and arc[1] % self.nodes != 0 and arc[1] - arc[0] < 2]

            for node in range(self.nodes):
                arcs.append((node, node+self.nodes))

            twodbn.addArcs(arcs)
            twodbn.generateCPTs()
            self.true_dbn=gdyn.unroll2TBN(twodbn,self.timesteps)


        if self.structure=='timesteps':
            #TODO: CPTs ARE GENERATED AFTER BUILD, I.E. THEY ARE NOT CONSISTENT OVER TIME. UNROLL DOESNT WORK FOR 
            #      BIGGER TIMESTEPS SO FIND A DIFFERENT WAY TO MAKE CPTs CONSISTENT 

            # Initialize true_dbn
            self.true_dbn = gum.BayesNet()
            # Create nodes
            all_nodes = []
            all_names = []

            names = [f"{chr(ord('`')+(x%26 + 1))}"*(np.ceil((x+1)/26).astype(int)) for x in range(self.nodes)]
            # print(names)
            for t in [f'{ti}' for ti in range(self.timesteps)]:
                for i in range(self.nodes):
                    name = names[i]+t
                    all_names.append(name)
                    all_nodes.append(self.true_dbn.add(gum.LabelizedVariable(name,name,self.values)))

            # Add causal arcs
            arcs = []
            causal_arcs = [arc for arc in list(itertools.combinations(all_nodes,2)) if arc[1] % self.nodes != 0 and arc[1] - arc[0] < 2]
            arcs += causal_arcs

            # Add time arcs (weighted to have more smaller steps)
            possible_times = [t for t in range(1, self.timesteps) if (self.timesteps-1) % t == 0]
            possible_skips = possible_times[::-1]
            for node in range(self.nodes):
                connections = random.choices(possible_times, k=1, weights=[t for t in possible_times])[0]
                # print(connections, possible_skips[possible_times.index(connections)])
                current = node
                for _ in range(1, connections+1):
                    time_arc=(current,current + self.nodes*possible_skips[possible_times.index(connections)])
                    arcs.append(time_arc)
                    # print(time_arc)
                    current = time_arc[1]
            
            self.true_dbn.addArcs(arcs)
            self.true_dbn.generateCPTs()

        if self.structure=='diamonds':
            arcs = []#[arc for arc in list(itertools.combinations(all_nodes,2)) if arc[1] % self.nodes != 0 and ((arc[1] %3==0 and arc[1] - arc[0] < 3) or ())]
            for t in range(2):
                for i in range(self.nodes):
                    node = i + self.nodes*t
                    x = (node % self.nodes) % 3
                    if x == 0:
                        if node + 2 < len(all_nodes) and node + 2 < self.nodes*(t+1):
                            arcs.append((node,node+1))
                            arcs.append((node,node+2))
                        elif node + 1 < len(all_nodes) and node + 1 < self.nodes*(t+1):
                            arcs.append((node,node+1))
                    elif x == 1:
                        if node + 2 < len(all_nodes) and node + 2 < self.nodes*(t+1):
                            arcs.append((node,node+2))
                    elif x == 2:
                        if node + 1 < len(all_nodes) and node + 1 < self.nodes*(t+1):
                            arcs.append((node,node+1))

            for node in range(self.nodes):
                arcs.append((node, node+self.nodes))

            twodbn.addArcs(arcs)
            twodbn.generateCPTs()
            self.true_dbn=gdyn.unroll2TBN(twodbn,self.timesteps)



        return self.true_dbn
    
    def generate_data(self,items=10000):
        self.train_items = items
        self.data,_ = gum.generateSample(self.true_dbn,items,None,False)
        self.data = self.data.reindex(sorted(self.data.columns, key=lambda x: (len(x), x[::-1], x[-1])), axis=1)
        self.data = self.data.astype('int')
        return self.data

    def learn_causal_structure(self):
        train = self.data.copy()[self.data.columns[:self.nodes]]

        discretizer = skbn.BNDiscretizer(defaultDiscretizationMethod='uniform',defaultNumberOfBins=5,discretizationThreshold=25)
        # Create nodes
        template = gum.BayesNet()
        for name in train:
            template.add(discretizer.createVariable(name, train[name]))
        # Set up learner
        self.causal_learner = gum.BNLearner(train, template)

        # TODO: CHOOSE BASED ON INPUT
        # Choose Algorithm and Score function
        self.causal_learner.useGreedyHillClimbing()
        self.causal_learner.useScoreBDeu()

        self.causal = self.causal_learner.learnBN()
        return self.causal
    
    def learn_time_structure(self):
        times = [0,1]
        train = self.data[[column for column in self.data.columns if int(column[-1]) in times]]

        discretizer = skbn.BNDiscretizer(defaultDiscretizationMethod='uniform',defaultNumberOfBins=5,discretizationThreshold=25)

        # Create nodes
        template = gum.BayesNet()
        for name in train:
            template.add(discretizer.createVariable(name, train[name]))

        # Set up learner
        self.time_learner = gum.BNLearner(train, template)

        # TODO: CHOOSE BASED ON INPUT
        # Choose Algorithm and Score function
        self.time_learner.useGreedyHillClimbing()
        self.time_learner.useScoreBDeu()

        # Force no back in time arcs
        for name1 in train:
            for name2 in train:
                if int(name1[-1]) > int(name2[-1]):
                    self.time_learner.addForbiddenArc(name1,name2)
                if int(name1[-1]) == int(name2[-1]):
                    self.time_learner.addForbiddenArc(name1,name2)

        self.time = self.time_learner.learnBN()
        return self.time
    
    def get_structure(self):
        # Initialize true_dbn
        self.learned = gum.BayesNet()
        # Create nodes
        all_nodes = []
        all_names = []

        names = [f"{chr(ord('`')+(x%26 + 1))}"*(np.ceil((x+1)/26).astype(int)) for x in range(self.nodes)]
        # print(names)
        for t in range(self.timesteps):
            for i in range(self.nodes):
                name = names[i]+f"{t}"
                all_names.append(name)
                all_nodes.append(self.learned.add(gum.LabelizedVariable(name,name,self.values)))

        arcs = []

        # Add causal structure and timestructure
        for t in range(self.timesteps): 
            for arc in self.causal.arcs():
                node1, node2 = self.causal_learner.nameFromId(arc[0])[:-1], self.causal_learner.nameFromId(arc[1])[:-1]
                arcs.append((node1+f"{t}",node2+f"{t}"))
            
            if t < self.timesteps-1:
                for arc in self.time.arcs():
                    node1, node2 = self.time_learner.nameFromId(arc[0])[:-1], self.time_learner.nameFromId(arc[1])[:-1]
                    arcs.append((node1+f"{t}",node2+f"{t+1}"))
        self.learned.addArcs(arcs)
        return self.learned
    
    def compare(self):
        gcmp=bnvsbn.GraphicalBNComparator(self.true_dbn,self.learned)
        scores = gcmp.skeletonScores()
        count = scores.pop("count")
        scores.update(count)
        
        self.parameters.update(scores)
        return self.parameters
    

def test_method(structure,nodes,timesteps,datapoints,values,method,scorefunction):
    # print(structure, nodes, timesteps, datapoints, method, scorefunction)
    generator = Generator(nodes=nodes, timesteps=timesteps, values=values, structure=structure, algorithm=method, score=scorefunction)
    generator.generate_DBN()

    scores = {}
    for _ in range(2):
        generator.generate_data(items=datapoints)
        generator.learn_causal_structure()
        generator.learn_time_structure()
        generator.get_structure()

        #TODO: Get Accuracy Learned BN
        score = {k:[v] for k,v in generator.compare().items()}
        if scores:
            for key, value in scores.items():
                scores[key] = value + score[key]
        else:
            scores = score

    return pd.DataFrame(scores)#.from_dict({k:[v] for k,v in scores.items()})



if __name__ == "__main__":
    structures = ['serie', 'islands']#, 'timesteps', 'diamonds']
    nodes = [4]#,6]
    timesteps = [3]#,5]
    datapoints = [100]#,150]
    values = [2]
    methods = ['temp']
    scorefunctions = ['temp']

    pool = mp.Pool(2)#mp.cpu_count())
    results = [pool.apply(test_method, args=(s, n, t, d, v, m, f)) 
               for s in structures 
               for n in nodes 
               for t in timesteps 
               for d in datapoints 
               for v in values 
               for m in methods 
               for f in scorefunctions]
    pool.close() 

    print(pd.concat(results, ignore_index=True))
