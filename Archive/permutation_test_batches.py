from tqdm import tqdm 
import itertools
from npeet import entropy_estimators as ee
import numpy as np
import pickle
import multiprocess
import sys
import argparse
import time
import pandas as pd
from math import comb

def compute_o_info(data):
    '''
    data is a pandas dataframe [num_observations, num_variables]

    Parameters
    ----------
    data : pandas dataframs
        Shape [num_observations, num_variables]

    Returns
    -------
    numpy.float64

    '''
    o_info = (len(data.columns) - 2)*ee.entropyd(data)
    for j,_ in (enumerate(data.columns)):
        #o_info += ee.entropyd(data.loc[:,data.columns == j])
        o_info += ee.entropyd(data.loc[:,data.columns == _])
        o_info -= ee.entropyd(data.loc[:,data.columns != _])
    
    return(o_info)

def permute_sample(i,batch,factor,columns,broken_sample, sema = None):
        
    dict_simulation_info = []
    for _,triplet in enumerate(itertools.combinations(columns, 3)): # undirected_triplets
        
        if (_ >= (i-1)*factor) and (_ < (i)*factor):
        

            o_info = compute_o_info(broken_sample[list(triplet)])
            dict_simulation_info.append({'triplet': triplet, 'O-info': o_info})

    df_simulation_info = pd.DataFrame(dict_simulation_info)
    #df_simulation_info.to_csv("YFS/Oinfo_permutation_test/Observed_Oinfo_set_"+str(batch)+"_part_"+str(i)+".csv")
    
    df_simulation_info.to_csv("Longitudinal Ageing Studies/UK/Oinfo_permutation_test/small/Observed_Oinfo_set_"+str(batch)+"_part_"+str(i)+".csv")
    
    if sema:
        sema.release()
    
    
def run_sims(batch = None):
    
    jobs = []

    # Number of threads we will use in parallel
    sema = multiprocess.Semaphore( int(multiprocess.cpu_count()-2))
    
    for batch in range(1000):
        #df = pd.read_csv('YFS/yfs_discrete_data_all_1105.csv') 
        df = pd.read_csv('Longitudinal Ageing Studies/UK/wave_6_elsa_Ps_Sp.csv')
        df = df.loc[:, df.columns != "Unnamed: 0"] 
        
        columns = df.columns
        Num_rows = df.shape[0]
        Num_vars = df.shape[1]
        N_size = 1000 # Sample size for bootstrap sample
        
        #broken_oinfo = np.empty((N_blocks,Num_vars,Num_vars))
    
        # P values -- need to break the MI for Null hypothesis
    
        bootstrap_sample = df #pd.DataFrame(df.values[np.random.randint(Num_rows, size=N_size)], columns=columns)
        
        factor = comb(len(columns),3)/int(multiprocess.cpu_count()-2)
        
        # Shuffle all the values
        broken_sample = bootstrap_sample.copy()
        for col in columns:
            broken_sample[col] = np.random.permutation(broken_sample[col])
        
        
            
        for i in range(int(multiprocess.cpu_count()-2) + 1):     
            sema.acquire()
            p = multiprocess.Process(target=permute_sample, args=(i,batch,factor,columns, broken_sample,sema)  )
            jobs.append(p)
            p.start()  
                    
        
    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
        

    print("All jobs finished")
    

if __name__ == '__main__':
    args = sys.argv
    print(args)
    print(args[1])

    run_sims(args[1])