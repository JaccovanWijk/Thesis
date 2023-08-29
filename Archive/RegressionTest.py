from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

import os
import pandas as pd
import numpy as np
import multiprocessing as mp


def parallel(num, regr, datapoints, pc1, weight1, above95, items_nodestates, ratio, meandegree, maxdegree, leaves, roots, constraint, items_arcs):
    best_alg = None
    best_fscore = 0

    for alg in ['algorithmMetric_3OFF2_constraint', 'algorithmMetric_HC_AIC','algorithmMetric_HC_BDEU', 'algorithmMetric_HC_BIC','algorithmMetric_HC_K2', 'algorithmMetric_HC_L2L','algorithmMetric_MIIC_constraint', 'algorithmMetric_MMHC_BDEU','algorithmMetric_MMHC_BIC', 'algorithmMetric_MMHC_K2','algorithmMetric_TABU_AIC', 'algorithmMetric_TABU_BDEU','algorithmMetric_TABU_BIC', 'algorithmMetric_TABU_K2','algorithmMetric_TABU_L2L']:
        X_predict = pd.DataFrame({  'items':[datapoints], 
                                    'arcs_ratio':[ratio], 
                                    'PC1':[pc1],
                                    'PCweight1':[weight1], 
                                    'PCAsAbove95':[above95], 
                                    'maxdegree/nodes':[maxdegree],
                                    'leaves/nodes':[leaves], 
                                    'roots/nodes':[roots], 
                                    'mean_constraint':[constraint], 
                                    'mean_degree':[meandegree],
                                    'items/nodestates':[items_nodestates], 
                                    'items/arcs':[items_arcs],
                                    'algorithmMetric_3OFF2_constraint':[int(alg == 'algorithmMetric_3OFF2_constraint')], 
                                    'algorithmMetric_HC_AIC':[int(alg == 'algorithmMetric_HC_AIC')],
                                    'algorithmMetric_HC_BDEU':[int(alg == 'algorithmMetric_HC_BDEU')], 
                                    'algorithmMetric_HC_BIC':[int(alg == 'algorithmMetric_HC_BIC')],
                                    'algorithmMetric_HC_K2':[int(alg == 'algorithmMetric_HC_K2')], 
                                    'algorithmMetric_HC_L2L':[int(alg == 'algorithmMetric_HC_L2L')],
                                    'algorithmMetric_MIIC_constraint':[int(alg == 'algorithmMetric_MIIC_constraint')], 
                                    'algorithmMetric_MMHC_BDEU':[int(alg == 'algorithmMetric_MMHC_BDEU')],
                                    'algorithmMetric_MMHC_BIC':[int(alg == 'algorithmMetric_MMHC_BIC')], 
                                    'algorithmMetric_MMHC_K2':[int(alg == 'algorithmMetric_MMHC_K2')],
                                    'algorithmMetric_TABU_AIC':[int(alg == 'algorithmMetric_TABU_AIC')], 
                                    #'algorithmMetric_TABU_BDEU':[int(alg == 'algorithmMetric_TABU_BDEU')],
                                    'algorithmMetric_TABU_BIC':[int(alg == 'algorithmMetric_TABU_BIC')], 
                                    'algorithmMetric_TABU_K2':[int(alg == 'algorithmMetric_TABU_K2')],
                                    'algorithmMetric_TABU_L2L':[int(alg == 'algorithmMetric_TABU_L2L')]})
        
        poly = PolynomialFeatures(degree=2)
        X_predict_poly = pd.DataFrame(poly.fit_transform(X_predict), columns=poly.get_feature_names_out(X_predict.columns))


        score = regr.predict(X_predict_poly)
        if score > best_fscore:
            best_alg = alg
            best_fscore = score

    pd.DataFrame.from_dict([{'items':datapoints,'PC1':pc1, 
                    'PCweight1':weight1, 
                    'PCAsAbove95':above95, 
                    'items/nodestates':items_nodestates,
                    'arcs_ratio':ratio,
                    'maxdegree/nodes':maxdegree,
                    'leaves/nodes':leaves, 
                    'roots/nodes':roots, 
                    'mean_constraint':constraint, 
                    'mean_degree':meandegree,
                    'items/arcs':items_arcs,
                    'algorithm':best_alg,
                    'fscore':best_fscore}]).to_csv(f"sensitivity/{num}.csv")
    # print(datapoints, pc1, weight1, above95, items_nodestates, ratio, meandegree, maxdegree, leaves, roots, constraint, items_arcs)
    return num






if __name__=="__main__":
    df = pd.read_csv("newOutputs/comparison.csv", index_col=0)

    def normalize(col):
        if max(col) > 1:
            return col/max(col)
        return col

    columns = ["items","nodes","timesteps","arcs_ratio","values","PC1","PC2","PC3","PC4","PCweight1","PCweight2","PCweight3","PCweight4","PCAsAbove95","maxdegree/nodes","roots/nodes","leaves/nodes","longestpath/nodes","mean_constraint","mean_nr_similarities/nodes","mean_degree","nodestates","arcs","items/nodestates","items/arcs"]
    df[columns] = df[columns].apply(normalize)

    def combine(row):
        return row["algorithm"]+"_"+row["scorefunction"]

    df = df[(df["algorithm"]!="TABU")|(df["scorefunction"]!="BDEU")]

    df["algorithmMetric"] = df.apply(combine, axis=1)
    df = df.drop(["algorithm","scorefunction"], axis=1)

    dfhot = pd.get_dummies(df, columns=["algorithmMetric"], dtype=int)

    columns = ['items', 'arcs_ratio', #'nodes', 'timesteps', 'values',  
       'PC1', #'PC2', #'PC3', 'PC4', 
        'PCweight1', #'PCweight2', #'PCweight3', 'PCweight4', 
        'PCAsAbove95', 
        'maxdegree/nodes',
       'leaves/nodes', 'roots/nodes',  #'longestpath/nodes', 
       'mean_constraint', #'mean_nr_similarities/nodes', 
        'mean_degree',
       'items/nodestates', 'items/arcs', #'nodestates', 'arcs', 
       'algorithmMetric_3OFF2_constraint', 'algorithmMetric_HC_AIC',
       'algorithmMetric_HC_BDEU', 'algorithmMetric_HC_BIC',
       'algorithmMetric_HC_K2', 'algorithmMetric_HC_L2L',
       'algorithmMetric_MIIC_constraint', 'algorithmMetric_MMHC_BDEU',
       'algorithmMetric_MMHC_BIC', 'algorithmMetric_MMHC_K2',
       'algorithmMetric_TABU_AIC', #'algorithmMetric_TABU_BDEU',
       'algorithmMetric_TABU_BIC', 'algorithmMetric_TABU_K2',
       'algorithmMetric_TABU_L2L']

    X_train, X_test, y_train, y_test = train_test_split(dfhot[columns], dfhot["fscore"], test_size = 0.2)

    poly = PolynomialFeatures(degree=2)
    X_train_poly, X_test_poly = pd.DataFrame(poly.fit_transform(X_train), columns=poly.get_feature_names_out(X_train.columns)), pd.DataFrame(poly.fit_transform(X_test), columns=poly.get_feature_names_out(X_test.columns))

    regr = XGBRegressor(n_estimators=500, max_depth=14, eta=0.1, subsample=0.7, colsample_bytree=0.8)#, colsample_bynode=0.8)
    regr.fit(X_train_poly, y_train)

    print(regr.score(X_test_poly, y_test))

    # regr = None
    # parameters = [(datapoints, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) 
    #                                 for datapoints in [100, 1000, 10000] ] 

    pc1s = np.linspace(min(dfhot["PC1"].unique()), max(dfhot["PC1"].unique()), 3)
    weights = np.linspace(min(dfhot["PCweight1"].unique()), max(dfhot["PCweight1"].unique()), 3)
    aboves = np.linspace(min(dfhot["PCAsAbove95"].unique()), max(dfhot["PCAsAbove95"].unique()), 3)
    itemnodestatess = np.linspace(min(dfhot["items/nodestates"].unique()), max(dfhot["items/nodestates"].unique()), 3)
    ratios = np.linspace(min(dfhot["arcs_ratio"].unique()), max(dfhot["arcs_ratio"].unique()), 3) 
    meandegrees =  np.linspace(min(dfhot["mean_degree"].unique()), max(dfhot["mean_degree"].unique()), 3) 
    maxdegrees = np.linspace(min(dfhot["maxdegree/nodes"].unique()), max(dfhot["maxdegree/nodes"].unique()), 3) 
    leavess = np.linspace(min(dfhot["leaves/nodes"].unique()), max(dfhot["leaves/nodes"].unique()), 3) 
    rootss = np.linspace(min(dfhot["roots/nodes"].unique()), max(dfhot["roots/nodes"].unique()), 3)
    constraintss = np.linspace(min(dfhot["mean_constraint"].unique()), max(dfhot["mean_constraint"].unique()), 3) 
    itemarcss = np.linspace(min(dfhot["items/arcs"].unique()), max(dfhot["items/arcs"].unique()), 3)
    parameters = [(datapoints, pc1, weight1, above95, items_nodestates, ratio, meandegree, maxdegree, leaves, roots, constraint, items_arcs) 
                                    for datapoints in [100, 1000, 10000] 
                                    for pc1 in pc1s
                                    for weight1 in weights
                                    for above95 in aboves
                                    for items_nodestates in itemnodestatess
                                    for ratio in ratios
                                    for meandegree in meandegrees
                                    for maxdegree in maxdegrees
                                    for leaves in leavess
                                    for roots in rootss
                                    for constraint in constraintss
                                    for items_arcs in itemarcss]
    
    params1 = [(num, regr, datapoints, pc1, weight1, above95, items_nodestates, ratio, meandegree, maxdegree, leaves, roots, constraint, items_arcs) for num, (datapoints, pc1, weight1, above95, items_nodestates, ratio, meandegree, maxdegree, leaves, roots, constraint, items_arcs) in enumerate(parameters)]
    done = os.listdir("sensitivity/")
    params2 = [tup for tup in params1 if f"{tup[0]}.csv" not in done]

    print("params generated")
    
    # params = [(257512, 100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)]
    with mp.Pool(mp.cpu_count()) as p:#5) as p:
        runs = p.starmap(parallel, params2)
        # runs = p.starmap(parallel, [(regr, dat, 0.09, 0.8, 0.75, 10, 2, 0.25, 0.75, 0.5, 0.5, 0.5, 0.3) for dat in [100,150]])
    # print(parallel((regr, params1[0])))

    print(len(runs))
    # df = pd.DataFrame.from_dict(runs)
    # df.to_csv("best_algorithms.csv")