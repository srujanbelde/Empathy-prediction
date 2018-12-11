from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV

def randomForestParamSelection(X,y,nFolds):
    rfc=RandomForestClassifier(random_state=42)
    param_grid = { 
    'n_estimators': list(range(3,25,3)),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : list(range(1,int(len(X.columns)/2),3)),
    'criterion' :['gini', 'entropy']
    }
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= nFolds)
    grid_search.fit(X,y)
    return grid_search.best_params_


def randomForest(X,y,nFolds):
    para = randomForestParamSelection(X,y,nFolds)
    finalClf = RandomForestClassifier(n_estimators = para.get('n_estimators'), max_features = para.get('max_features'), max_depth = para.get('max_depth'), criterion = para.get('criterion'))
    return finalClf
    