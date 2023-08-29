import pyAgrum as gum
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle as pkl


def main():
    files = os.listdir("outputs/scores")

    df = pd.DataFrame({"items":[],"nodes":[], "timesteps":[], "values":[], "algorithm":[], "scorefunction":[], "fscore":[], "PC1":[], "PC2":[], "PC3":[], "PC4":[]})
    # for file in np.random.choice(files,10):
    for file in files:
        structure, rep = file[9:-4].split("_")
        bn = gum.BayesNet()
        bn.loadBIFXML(f"outputs/structures/structure{structure}_{rep}.bifxml")
        
        for _,row in pd.read_csv(f"outputs/scores/structure{structure}_{rep}.csv").iterrows():
        
            data,_ = gum.generateSample(bn,row["items"],None,False)
            data = data.reindex(sorted(data.columns, key=lambda x: (len(x), int(x.split(".")[1]), x.split(".")[0])), axis=1)
            data = data.astype('int')

            cov = data.cov()

            pca = PCA()
            _ = pca.fit_transform(cov)
            PC_components = np.arange(pca.n_components_) + 1
            ratios = pca.explained_variance_ratio_
            df = pd.concat([df,pd.DataFrame({"items":[row["items"]], "nodes":[row["nodes"]], "timesteps":[row["timesteps"]], "values":[row["values"]], "algorithm":[row["algorithm"]], "scorefunction":[row["score"]], "fscore":[row["fscore"]], "PC1":[ratios[0]], "PC2":[ratios[1]], "PC3":[ratios[2]], "PC4":[ratios[3]]})])
    print("df done")
    df['algorithmi'] = df['algorithm'].map({'HC': 0, 'TABU':1, '3OFF2':2, 'MIIC':3, 'MMHC':4})
    df['scorefunctioni'] = df['scorefunction'].map({'BDEU': 0, 'AIC':1, 'BIC':2, 'K2':3, 'L2L':4})

    X = df[["algorithmi","scorefunctioni","PC1","PC2","PC3","PC4"]]
    y = df["fscore"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    X = df[["items","algorithmi","scorefunctioni","PC1","PC2","PC3","PC4"]]
    y = df["fscore"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_val)
    mse = metrics.mean_squared_error(y_val, y_pred)
    rmse = mse**0.5

    importances = regr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regr.estimators_], axis=0)
    # forest_importances = pd.Series(importances, index=X.columns)

    output = {"rmse":rmse, "importances":importances, "std":std, "decisions?":regr}

    with open('classify_outputs.pkl', 'wb') as f:
        pkl.dump(output, f)

if __name__ == "__main__":
    main()