from sklearn import svm
from sklearn.model_selection import cross_val_score,GridSearchCV

def SVMkernal(X,y):
    """
    acc = 0
    deg = 0
    para = svc_param_selection(X,y,5)
    for i in range(1,20):
        clf = svm.SVC(C = para.get('C'),kernel='poly', degree=i, gamma = para.get('gamma')) 
        scores = cross_val_score(clf, X, y, cv=3)
        if(scores.mean() > acc):
            acc = scores.mean()
            deg = i
    finalClf = svm.SVC(C = para.get('C'),kernel='poly', degree = deg, gamma = para.get('gamma'))
    return finalClf
    """
    
    para = svc_param_selection(X,y,5)
    finalClf = svm.SVC(probability=True,C = para.get('C'),kernel='poly', degree = para.get('degree'), gamma = para.get('gamma'))
    return finalClf

def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.01, 0.1, 1,5]
    degrees = list(range(1,12))
    param_grid = {'C': Cs, 'gamma' : gammas, 'degree' : degrees}
    grid_search = GridSearchCV(svm.SVC(kernel='poly'), param_grid, cv=nfolds)
    grid_search.fit(X,y)
    return grid_search.best_params_