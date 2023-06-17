import pandas as pd
import numpy as np
import itertools
import sys
import pyAgrum as gum
import pyAgrum.lib.dynamicBN as gdyn
import pyAgrum.skbn as skbn
import pyAgrum.lib.bn_vs_bn as gcm
import random
import multiprocessing as mp
import pyAgrum.lib.bn_vs_bn as bnvsbn
import pickle as pkl

class Generator:
    def __init__(self, items=1000, nodes=4, timesteps=3, common_effects=0.5, common_causes=0.5, islands=1, values=2, algorithm="HC", score="BDEU"): #structure='serie',
        self.items = items
        self.nodes = nodes
        self.timesteps = timesteps
        # self.structure = structure
        self.relative_common_effects = common_effects
        self.relative_common_causes = common_causes
        self.islands = islands
        self.values = values
        self.algorithm = algorithm
        self.scorefunction = score
        return 

    def generate_structure(self):
        # Initialize true_dbn
        twodbn = gum.BayesNet()
        # Create nodes
        all_nodes = []
        all_names = []

        names = [f"{chr(ord('`')+(x%26 + 1))}"*(np.ceil((x+1)/26).astype(int)) for x in range(self.nodes)]
        # print(names)
        for t in ['0','t']:
            for i in range(self.nodes):
                name = names[i]+"."+t
                all_names.append(name)
                all_nodes.append(twodbn.add(gum.LabelizedVariable(name,name,self.values)))

        split_nodes = np.array(np.split(np.array(range(self.nodes)),self.islands))


        # if self.common_effects > self.nodes - 2*self.islands:
        #     self.common_effects = self.nodes - 2*self.islands
        # if self.common_causes > self.nodes - 2*self.islands:
        #     self.common_causes = self.nodes - 2*self.islands
        self.absolute_common_effects = int(np.floor((self.nodes - 2*self.islands)*self.relative_common_effects))
        self.absolute_common_causes = int(np.floor((self.nodes - 2*self.islands)*self.relative_common_causes))

        effects = np.random.choice([x for x in range(self.nodes) if x not in split_nodes[:,:2]], self.absolute_common_effects, replace=False)
        causes = np.random.choice([x for x in range(self.nodes) if x not in split_nodes[:,-2:]], self.absolute_common_causes, replace=False)

        arcs = []#[arc for arc in list(itertools.combinations(all_nodes,2)) if arc[1] % nodes != 0 and arc[1] - arc[0] < 2]
        for split in split_nodes:
            for node in split:
                node = int(node)
                if node in effects:
                    if random.random()< 0.5:
                        arcs += [(node-2,node), (node-1,node), (node+self.nodes-2,node+self.nodes), (node+self.nodes-1,node+self.nodes)]
                    else:
                        arcs += [(node-2,node), (node-1,node), (node-2, node-1), (node+self.nodes-2,node+self.nodes), (node+self.nodes-1,node+self.nodes), (node+self.nodes-2,node+self.nodes-1),]
                if node in causes:
                    if random.random()< 0.5:
                        arcs += [(node,node+1), (node,node+2), (node+self.nodes,node+self.nodes+1), (node+self.nodes,node+self.nodes+2)]
                    else:
                        arcs += [(node,node+1), (node,node+2), (node+1,node+2), (node+self.nodes,node+self.nodes+1), (node+self.nodes,node+self.nodes+2), (node+self.nodes+1,node+self.nodes+2)]
                if node not in causes and node not in split_nodes[:,-1]:
                    arcs += [(node,node+1), (node+self.nodes,node+self.nodes+1)]       

        for node in range(self.nodes):
            arcs.append((node, node+self.nodes))
            
        twodbn.addArcs(set(arcs))
        return twodbn
    
    def generate_DBN(self, twodbn):
        twodbn.generateCPTs()
        self.true_dbn=gdyn.unroll2TBN(twodbn,self.timesteps)
        return self.true_dbn



    #     return self.true_dbn


    # def generate_DBN(self):
    #     # Initialize true_dbn
    #     twodbn = gum.BayesNet()
    #     # Create nodes
    #     all_nodes = []
    #     all_names = []

    #     names = [f"{chr(ord('`')+(x%26 + 1))}"*(np.ceil((x+1)/26).astype(int)) for x in range(self.nodes)]
    #     # print(names)
    #     for t in ['0','t']:
    #         for i in range(self.nodes):
    #             name = names[i]+"."+t
    #             all_names.append(name)
    #             all_nodes.append(twodbn.add(gum.LabelizedVariable(name,name,self.values)))

    #     if self.structure=='serie':
    #         #OC Stands for one causal structure!!!
    #         arcs = [arc for arc in list(itertools.combinations(all_nodes,2)) if arc[1] % self.nodes != 0 and arc[1] - arc[0] < 2]

    #         for node in range(self.nodes):
    #             arcs.append((node, node+self.nodes))

    #         twodbn.addArcs(arcs)
    #         twodbn.generateCPTs()
    #         self.true_dbn=gdyn.unroll2TBN(twodbn,self.timesteps)

    #     if self.structure=='islands':
    #         #OC Stands for one causal structure!!!
    #         arcs = [arc for arc in list(itertools.combinations(all_nodes,2)) if arc[1] % int(self.nodes/2) != 0 and arc[1] % self.nodes != 0 and arc[1] - arc[0] < 2]

    #         for node in range(self.nodes):
    #             arcs.append((node, node+self.nodes))

    #         twodbn.addArcs(arcs)
    #         twodbn.generateCPTs()
    #         self.true_dbn=gdyn.unroll2TBN(twodbn,self.timesteps)


    #     if self.structure=='timesteps':
    #         #TOdDO: CPTs ARE GENERATED AFTER BUILD, I.E. THEY ARE NOT CONSISTENT OVER TIME. UNROLL DOESNT WORK FOR 
    #         #      BIGGER TIMESTEPS SO FIND A DIFFERENT WAY TO MAKE CPTs CONSISTENT 

    #         # Initialize true_dbn
    #         self.true_dbn = gum.BayesNet()
    #         # Create nodes
    #         all_nodes = []
    #         all_names = []

    #         names = [f"{chr(ord('`')+(x%26 + 1))}"*(np.ceil((x+1)/26).astype(int)) for x in range(self.nodes)]
    #         # print(names)
    #         for t in [f'{ti}' for ti in range(self.timesteps)]:
    #             for i in range(self.nodes):
    #                 name = names[i]+"."+t
    #                 all_names.append(name)
    #                 all_nodes.append(self.true_dbn.add(gum.LabelizedVariable(name,name,self.values)))

    #         # Add causal arcs
    #         arcs = []
    #         causal_arcs = [arc for arc in list(itertools.combinations(all_nodes,2)) if arc[1] % self.nodes != 0 and arc[1] - arc[0] < 2]
    #         arcs += causal_arcs

    #         # Add time arcs (weighted to have more smaller steps)
    #         possible_times = [t for t in range(1, self.timesteps) if (self.timesteps-1) % t == 0]
    #         possible_skips = possible_times[::-1]
    #         for node in range(self.nodes):
    #             connections = random.choices(possible_times, k=1, weights=[t for t in possible_times])[0]
    #             # print(connections, possible_skips[possible_times.index(connections)])
    #             current = node
    #             for _ in range(1, connections+1):
    #                 time_arc=(current,current + self.nodes*possible_skips[possible_times.index(connections)])
    #                 arcs.append(time_arc)
    #                 # print(time_arc)
    #                 current = time_arc[1]
            
    #         self.true_dbn.addArcs(arcs)
    #         self.true_dbn.generateCPTs()

    #     if self.structure=='diamonds':
    #         arcs = []#[arc for arc in list(itertools.combinations(all_nodes,2)) if arc[1] % self.nodes != 0 and ((arc[1] %3==0 and arc[1] - arc[0] < 3) or ())]
    #         for t in range(2):
    #             for i in range(self.nodes):
    #                 node = i + self.nodes*t
    #                 x = (node % self.nodes) % 3
    #                 if x == 0:
    #                     if node + 2 < len(all_nodes) and node + 2 < self.nodes*(t+1):
    #                         arcs.append((node,node+1))
    #                         arcs.append((node,node+2))
    #                     elif node + 1 < len(all_nodes) and node + 1 < self.nodes*(t+1):
    #                         arcs.append((node,node+1))
    #                 elif x == 1:
    #                     if node + 2 < len(all_nodes) and node + 2 < self.nodes*(t+1):
    #                         arcs.append((node,node+2))
    #                 elif x == 2:
    #                     if node + 1 < len(all_nodes) and node + 1 < self.nodes*(t+1):
    #                         arcs.append((node,node+1))

    #         for node in range(self.nodes):
    #             arcs.append((node, node+self.nodes))

    #         twodbn.addArcs(arcs)
    #         twodbn.generateCPTs()
    #         self.true_dbn=gdyn.unroll2TBN(twodbn,self.timesteps)



    #     return self.true_dbn
    
    def generate_data(self,items=10000):
        self.train_items = items
        self.data,_ = gum.generateSample(self.true_dbn,items,None,False)
        self.data = self.data.reindex(sorted(self.data.columns, key=lambda x: (int(x.split(".")[1]), x.split(".")[0])), axis=1)
        self.data = self.data.astype('int')
        return self.data

    def learn_causal_structure(self, subpoints):
        train = self.data.copy()[self.data.columns[:self.nodes]].sample(subpoints)

        discretizer = skbn.BNDiscretizer(defaultDiscretizationMethod='uniform',defaultNumberOfBins=5,discretizationThreshold=25)
        # Create nodes
        template = gum.BayesNet()
        for name in train:
            template.add(discretizer.createVariable(name, train[name]))
        # Set up learner
        self.causal_learner = gum.BNLearner(train, template)

        # Choose Algorithm and Score function
        if self.algorithm == 'HC':
            self.causal_learner.useGreedyHillClimbing()
        elif self.algorithm == 'TABU':
            self.causal_learner.useLocalSearchWithTabuList()

        if self.scorefunction == 'BDEU':
            self.causal_learner.useScoreBDeu()
        elif self.scorefunction == 'AIC':
            self.causal_learner.useScoreAIC()
        elif self.scorefunction == 'BIC':
            self.causal_learner.useScoreBIC()

        self.causal = self.causal_learner.learnBN()
        return self.causal
    
    def learn_time_structure(self, subpoints, times=2):
        self.times = times
        train = self.data[[column for column in self.data.columns if int(column.split(".")[1]) < self.times]].sample(subpoints)

        discretizer = skbn.BNDiscretizer(defaultDiscretizationMethod='uniform',defaultNumberOfBins=5,discretizationThreshold=25)

        # Create nodes
        template = gum.BayesNet()
        for name in train:
            template.add(discretizer.createVariable(name, train[name]))

        # Set up learner
        self.time_learner = gum.BNLearner(train, template)

        # Choose Algorithm and Score function
        if self.algorithm == 'HC':
            self.time_learner.useGreedyHillClimbing()
        elif self.algorithm == 'TABU':
            self.time_learner.useLocalSearchWithTabuList()

        if self.scorefunction == 'BDEU':
            self.time_learner.useScoreBDeu()
        elif self.scorefunction == 'AIC':
            self.time_learner.useScoreAIC()

        # Force no back in time arcs
        for name1 in train:
            for name2 in train:
                if int(name1.split(".")[1]) > int(name2.split(".")[1]):
                    self.time_learner.addForbiddenArc(name1,name2)
                if int(name1.split(".")[1]) == int(name2.split(".")[1]):
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
                name = names[i]+f".{t}"
                all_names.append(name)
                all_nodes.append(self.learned.add(gum.LabelizedVariable(name,name,self.values)))

        arcs = []

        # Add causal structure and timestructure
        for t in range(self.timesteps): 
            for arc in self.causal.arcs():
                node1, node2 = self.causal_learner.nameFromId(arc[0]).split(".")[0], self.causal_learner.nameFromId(arc[1]).split(".")[0]
                arcs.append((node1+f".{t}",node2+f".{t}"))
        
        for t in range(0, self.timesteps, self.times-1):
            if t < self.timesteps-1:# and t % self.times-1 == 0:
                for arc in self.time.arcs():
                    # print(arc)
                    node1, node2 = self.time_learner.nameFromId(arc[0]).split(".")[0], self.time_learner.nameFromId(arc[1]).split(".")[0]
                    time1, time2 = int(self.time_learner.nameFromId(arc[0]).split(".")[1]) + t, int(self.time_learner.nameFromId(arc[1]).split(".")[1]) + t
                    if time2 < self.timesteps:
                        arcs.append((node1+f".{time1}",node2+f".{time2}"))
        self.learned.addArcs(arcs)
        return self.learned
    
    def compare(self, subpoints):
        # WHAT DO SCORES MEAN: https://pyagrum.readthedocs.io/en/latest/pyAgrum.lib.html#bn-vs-bn 
        gcmp=bnvsbn.GraphicalBNComparator(self.true_dbn,self.learned)

        structure_scores = {"items":subpoints, "nodes":self.nodes, "timesteps":self.timesteps, "values":self.values, "causes":self.relative_common_causes, "effects":self.relative_common_effects, "algorithm":self.algorithm, "score":self.scorefunction}
        structure_scores.update(gcmp.skeletonScores())
        count = structure_scores.pop("count")
        structure_scores.update(count)

        parameter_scores = {"items":subpoints, "nodes":self.nodes, "timesteps":self.timesteps, "values":self.values, "causes":self.relative_common_causes, "effects":self.relative_common_effects, "algorithm":self.algorithm, "score":self.scorefunction}
        parameter_scores.update(gcmp.scores())
        count = parameter_scores.pop("count")
        parameter_scores.update(count)
        return structure_scores, parameter_scores
    

def test_method(num,common_causes, common_effects,nodes,timesteps,datapoints,values,method,scorefunction, sema=None):

    structure_scores = {}
    parameter_scores = {}

    generator = Generator(nodes=nodes, timesteps=timesteps, values=values, common_effects=common_effects, common_causes=common_causes, algorithm=method, score=scorefunction)
    twodbn = generator.generate_structure()
    with open(f"/gpfs/home5/jvwijk/outputs/structures/structure{num}.pkl", 'wb') as f:
        pkl.dump(twodbn, f)
    for _ in range(4):#1000):
        generator.generate_DBN(twodbn)
        generator.generate_data(items=datapoints)
        for subpoints in [int(np.ceil(datapoints/10*i)) for i in range(1,11)]:
            generator.learn_causal_structure(subpoints)
            generator.learn_time_structure(subpoints)
            generator.get_structure()

    #         score = {k:[v] for k,v in generator.compare(subpoints).items()}
            structure_score, parameter_score = generator.compare(subpoints)
            structure_score = {k:[v] for k,v in structure_score.items()}
            parameter_score = {k:[v] for k,v in parameter_score.items()}

            if structure_scores:
                for key, value in structure_scores.items():
                    structure_scores[key] = value + structure_score[key]
            else:
                structure_scores = structure_score
                
            if parameter_scores:
                for key, value in parameter_scores.items():
                    parameter_scores[key] = value + parameter_score[key]
            else:
                parameter_scores = parameter_score

    # return pd.DataFrame(scores)#.from_dict({k:[v] for k,v in scores.items()})
    pd.DataFrame(structure_scores).to_csv(f"/gpfs/home5/jvwijk/outputs/structure_scores/structure{num}.csv")
    pd.DataFrame(parameter_scores).to_csv(f"/gpfs/home5/jvwijk/outputs/parameter_scores/structure{num}.csv")

    if sema:
        sema.release()

def run_sims():
    jobs = []
    cpus = int(mp.cpu_count() - 2)
    sema = mp.Semaphore(cpus)

    nodes = [4]#,5,6,7,8,10,15,20]
    timesteps = [1]#,2,3,4,5,6,10]
    datapoints = [10000]#0]
    values = [2]#, 3, 4]
    methods = ['HC']#, 'TABU']
    scorefunctions = ['BDEU']#, 'AIC', 'BIC']
    #TODO: WHAT PARAMETER LEARNING ALGORITHMS
    
    common_effects = [0, 0.25, 0.5, 0.75, 1]
    common_causes = [0, 0.25, 0.5, 0.75, 1]

    parameters = [(cc, ce, n, t, d, v, m, f) for cc in common_causes for ce in common_effects for n in nodes for t in timesteps for d in datapoints for v in values for m in methods for f in scorefunctions]

    for i, params in enumerate(parameters):
        batches = 2#1000
        for batch in range(batches):    
            cc, ce, n, t, d, v, m, f = params
            sema.acquire()
            p = mp.Process(target=test_method, args=(batch + i*batches,cc,ce,n,t,d,v,m,f, sema)) 
            jobs.append(p)
            p.start()

    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
        

    print("All jobs finished")


if __name__ == "__main__":
    run_sims()
