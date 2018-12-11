from sklearn import tree
from sklearn.model_selection import cross_val_score


def DT(X,y): 
    acc = 0
    depth = 0
    for i in range(1,20):
        clf = tree.DecisionTreeClassifier(max_depth = i)
        scores = cross_val_score(clf, X, y, cv=10)
        if(scores.mean() > acc):
            acc = scores.mean()
            depth = i
    finalClf = tree.DecisionTreeClassifier(max_depth = depth)
    return finalClf