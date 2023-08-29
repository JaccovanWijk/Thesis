import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
from multiprocessing import Pool, cpu_count
import time

from pgmpy.estimators import BDeuScore, K2Score, BicScore, MmhcEstimator, HillClimbSearch

class Generator:
    def __init__(self, items=1000, nodes=4, timesteps=3, arcs_ratio=1.5, values=2, algorithm="HC", score="BDEU"): #structure='serie', common_effects=0.5, common_causes=0.5, 
        self.items = items
        self.nodes = nodes
        self.timesteps = timesteps
        # self.structure = structure
        # self.relative_common_effects = common_effects
        # self.relative_common_causes = common_causes
        self.arcs_ratio = arcs_ratio
        self.values = values
        self.algorithm = algorithm
        self.scorefunction = score
        return 

    def generate_structure(self):
        all_names = []

        names = [f"{chr(ord('`')+(x%26 + 1))}"*(np.ceil((x+1)/26).astype(int)) for x in range(self.nodes)]
        # print(names)
        for t in ['0','t']:
            for i in range(self.nodes):
                name = names[i]+"."+t
                all_names.append(name)

        twodbn = gum.randomBN(n=self.nodes, names=all_names[:self.nodes], ratio_arc=self.arcs_ratio)

        for name in sorted(twodbn.names(), key=lambda x: twodbn.ids([x])):
            name = name.split(".")[0] + ".t"
            twodbn.add(gum.LabelizedVariable(name,name,2))
            
        arcs = []
        for arc in twodbn.arcs():
            arcs.append((arc[0]+self.nodes,arc[1]+self.nodes))
            
        for node in range(self.nodes):
            arcs.append((node, node+self.nodes))

        twodbn.addArcs(set(arcs))
        return twodbn
    
    def generate_DBN(self, twodbn):
        twodbn.generateCPTs()
        self.true_dbn=gdyn.unroll2TBN(twodbn,self.timesteps)
        return self.true_dbn

    
    def generate_data(self,items=10000):
        self.train_items = items
        self.data,_ = gum.generateSample(self.true_dbn,items,None,False)
        self.data = self.data.reindex(sorted(self.data.columns, key=lambda x: (len(x), int(x.split(".")[1]), x.split(".")[0])), axis=1)
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

        if self.algorithm == 'MMHC':
            if self.scorefunction == 'BDEU':
                scorefunction = BDeuScore(train)
            elif self.scorefunction == 'BIC':
                scorefunction = K2Score(train)
            elif self.scorefunction == 'K2':
                scorefunction = BicScore(train)
            
            est = MmhcEstimator(train)
            edges = est.estimate(scoring_method=scorefunction).edges()

            self.causal = gum.BayesNet()
            all_names = []
            all_nodes = []
            names = train.columns
            
            for i in range(len(names)):
                name = names[i]
                all_names.append(name)
                all_nodes.append(self.causal.add(gum.LabelizedVariable(name,name,2)))

            arcs=[]
            for arc in edges:
                node1, node2 = arc[0],arc[1]
                arcs.append((node1,node2))
            self.causal.addArcs(arcs)            
            
        else:
            # Choose Algorithm and Score function
            if self.algorithm == 'HC':
                self.causal_learner.useGreedyHillClimbing()
            elif self.algorithm == 'TABU':
                self.causal_learner.useLocalSearchWithTabuList()
            elif self.algorithm == '3OFF2':
                self.causal_learner.use3off2()
            elif self.algorithm == 'MIIC':
                self.causal_learner.useMIIC()

            if self.scorefunction == 'BDEU':
                self.causal_learner.useScoreBDeu()
            elif self.scorefunction == 'AIC':
                self.causal_learner.useSmoothingPrior()
                self.causal_learner.useScoreAIC()
            elif self.scorefunction == 'BIC':
                self.causal_learner.useSmoothingPrior()
                self.causal_learner.useScoreBIC()
            elif self.scorefunction == 'K2':
                self.causal_learner.useSmoothingPrior()
                self.causal_learner.useScoreK2()
            elif self.scorefunction == 'L2L':
                self.causal_learner.useSmoothingPrior()
                self.causal_learner.useScoreLog2Likelihood()

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

        if self.algorithm == 'MMHC':
            if self.scorefunction == 'BDEU':
                scorefunction = BDeuScore(train)
            elif self.scorefunction == 'BIC':
                scorefunction = K2Score(train)
            elif self.scorefunction == 'K2':
                scorefunction = BicScore(train)
            

            # Min max
            est = MmhcEstimator(train)
            skel = est.mmpc(0.01)

            # Hill Climb
            hc = HillClimbSearch(train)
            model = hc.estimate(
                scoring_method=scorefunction,
                # white_list=[edge for edge in skel.to_directed().edges() if int(edge[0].split(".")[1]) < int(edge[1].split(".")[1])],
                black_list=[edge for edge in skel.to_directed().edges() if int(edge[0].split(".")[1]) >= int(edge[1].split(".")[1])],
                show_progress=False
            )
            edges = model.edges()

            self.time = gum.BayesNet()
            all_names = []
            all_nodes = []
            names = train.columns
            
            for i in range(len(names)):
                name = names[i]
                all_names.append(name)
                all_nodes.append(self.time.add(gum.LabelizedVariable(name,name,2)))

            arcs=[]
            for arc in edges:
                node1, node2 = arc[0],arc[1]
                if int(node1.split(".")[1]) + 1 == int(node2.split(".")[1]):
                    arcs.append((node1,node2))
            self.time.addArcs(arcs)            
            
        else:

            self.time_learner.useSmoothingPrior()
            # Choose Algorithm and Score function
            if self.algorithm == 'HC':
                self.time_learner.useGreedyHillClimbing()
            elif self.algorithm == 'TABU':
                self.time_learner.useLocalSearchWithTabuList()
            elif self.algorithm == '3OFF2':
                self.time_learner.use3off2()
            elif self.algorithm == 'MIIC':
                self.time_learner.useMIIC()

            if self.scorefunction == 'BDEU':
                self.time_learner.useScoreBDeu()
            elif self.scorefunction == 'AIC':
                self.time_learner.useSmoothingPrior()
                self.time_learner.useScoreAIC()
            elif self.scorefunction == 'BIC':
                self.time_learner.useSmoothingPrior()
                self.time_learner.useScoreBIC()
            elif self.scorefunction == 'K2':
                self.time_learner.useSmoothingPrior()
                self.time_learner.useScoreK2()
            elif self.scorefunction == 'L2L':
                self.time_learner.useSmoothingPrior()
                self.time_learner.useScoreLog2Likelihood()

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
        
        scores = {"items":subpoints, "nodes":self.nodes, "timesteps":self.timesteps, "arcs_ratio":self.arcs_ratio, "values":self.values, "algorithm":self.algorithm, "scorefunction":self.scorefunction} #"causes":self.relative_common_causes, "effects":self.relative_common_effects,
        scores.update(gcmp.scores())
        count = scores.pop("count")
        scores.update(count)
        return scores
    

def test_method(num,arcs_ratio,nodes,timesteps, datapoints,values, sema=None):
    if method == 'MMHC' and (scorefunction=='AIC' or scorefunction=="L2L"):
        if sema:
            sema.release()
    else:
        generator = Generator(nodes=nodes, timesteps=timesteps, values=values, arcs_ratio=arcs_ratio, algorithm=method, score=scorefunction)
        twodbn = generator.generate_structure()
    
        for rep in range(30):
            scores = {}
            true_dbn = generator.generate_DBN(twodbn)
            
            true_dbn.saveBIFXML(f"/gpfs/home5/jvwijk/outputs/GT_structures/structure{num}_{rep}.bifxml")
            
            generator.generate_data(items=datapoints)
            
            for subpoints in [100,500]+[int(np.ceil(datapoints/10*i)) for i in range(1,11)]:
                generator.learn_causal_structure(subpoints)
                generator.learn_time_structure(subpoints)
                
                learned = generator.get_structure()
                learned.saveBIFXML(f"/gpfs/home5/jvwijk/outputs/first_structures/structure{num}_{rep}_{subpoints}.bifxml")

    
        #         score = {k:[v] for k,v in generator.compare(subpoints).items()}
                score = generator.compare(subpoints)
                score = {k:[v] for k,v in score.items()}
                    
                if scores:
                    for key, value in scores.items():
                        scores[key] = value + score[key]
                else:
                    scores = score
    
            pd.DataFrame(scores).to_csv(f"/gpfs/home5/jvwijk/outputs/scores/structure{num}_{rep}.csv")
    
        if sema:
            sema.release()
        # return num

def run_sims():
    jobs = []
    cpus = int(mp.cpu_count() - 2)
    sema = mp.Semaphore(cpus)

    nodes = [4,6,8,10]
    timesteps = [1,2,3]
    datapoints = [10000]
    values = [2, 4]
    methods = ['HC', 'TABU', 'MMHC', '3OFF2', 'MIIC']
    scorefunctions = ['BDEU', 'AIC', 'BIC', 'K2', 'L2L']
    
    arcs_ratio = [0.999999, 1.25, 1.5, 1.75, 2.]

    structure_parameters = [(r, n, t, d, v) for r in arcs_ratio for n in nodes for t in timesteps for d in datapoints for v in values]
    
    batches = 30
    for i, struc_params in enumerate(structure_parameters):
        for batch in range(batches):   
            r, n, t, d, v = struc_params
            sema.acquire()
            # TODO: SELECT BEST PERFORMING ALGORITHM AND SCORE METRIC
            p = mp.Process(target=test_method, args=(i+batch*len(structure_parameters),r,n,t,d,v,"TABU","K2", sema, True)) 
            jobs.append(p)
            p.start()











    # parameters = [(r, n, t, d, v, m, f) for r in arcs_ratio for n in nodes for t in timesteps for d in datapoints for v in values for m in methods for f in scorefunctions]

    # batches = 30
    # for i, params in enumerate(parameters):
    #     for batch in range(batches):   
    #         r, n, t, d, v, m, f = params
    #         sema.acquire()
    #         p = mp.Process(target=test_method, args=(i+batch*len(parameters),r,n,t,d,v,m,f, sema)) 
    #         jobs.append(p)
    #         p.start()

    # print("Number of jobs: " + str(len(jobs)))    
    # count = 0
    # for p in jobs:
    #     count += 1
    #     p.join()
    #     print("jobs done...", count)
        

    # print("All jobs finished")


if __name__ == "__main__":
    run_sims()
    # start = time.time()
    # test_method(0,0.9999,4,2,1,10000,2,'HC','BDEU')
    # print("took", time.time() - start)
