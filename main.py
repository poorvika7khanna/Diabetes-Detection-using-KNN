import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dist(x1, x2):
    return np.sqrt(sum((x1 - x2)**2))


def knn(X, Y, queryPoint, k=5):
    vals = []
    for i in range(X.shape[0]):
        distance = dist(queryPoint, X[i])
        vals.append((distance, Y[i]))
    vals = sorted(vals)
    vals = np.array(vals[:k], dtype=int)
    newval = np.unique(vals[:, 1], return_counts=True)
    ind = newval[1].argmax()
    pred = newval[0][ind]
    return pred


xtrain = pd.read_csv("Diabetes_XTrain.csv")
ytrain = pd.read_csv("Diabetes_YTrain.csv")
x = xtrain.values
y = ytrain.values.reshape((-1,))
# classes = np.unique(y, return_counts=True)
# plt.style.use('seaborn')
# plt.bar(classes[0], classes[1], width=0.25)
# plt.show()
xtest = pd.read_csv("Diabetes_Xtest.csv")
tx = xtest.values
solution = []
for i in range(tx.shape[0]):
    pred = knn(x, y, tx[i, :], 13)
    solution.append(pred)
ytest = pd.DataFrame(solution, columns=['Outcome'])
ytest.to_csv("Diabetes_Ytest.csv", index=False)
