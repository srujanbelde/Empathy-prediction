from sklearn.naive_bayes import MultinomialNB,GaussianNB

def gaussianNB(X,Y):
    clf = GaussianNB()
    clf.fit(X, Y)
    return clf