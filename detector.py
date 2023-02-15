from sklearn.preprocessing import OneHotEncoder
import numpy as np

def detector(trainSet, liveSet, parameters):
    feature1 = trainSet[trainSet.columns[0]].values
    feature2 = liveSet[liveSet.columns[0]].values
    enc = OneHotEncoder()
    a = enc.fit_transform(feature1.reshape(-1,1)).toarray()
    b = enc.fit_transform(feature2.reshape(-1,1)).toarray()
    l_inf = np.linalg.norm(np.abs(a.sum(axis=0)/a.shape[0] - b.sum(axis=0)/b.shape[0]), ord=np.inf)
    threshold = float (parameters.get("threshold", 0.10))
    return int(l_inf < threshold), l_inf
