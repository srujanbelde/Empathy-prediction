from imports import *

print("Reading Data from the CSV file and encoding it")
df1 = pd.read_csv('responses.csv')

df1 = pp.encoder(df1)
print("Encoding Finished")

y = df1[['Empathy']]
X = df1.drop(['Empathy'], axis=1)

print("Removing the Irrelavent Features")
X = pp.removeIrrelaventFeatures(X,y)

print("Removing Redundant Features")
X = pp.removeRedundantFeatues(X)

print("Normalizing the Data")
X = pp.Normalization(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.sort_index()
X_test = X_test.sort_index()
y_train = y_train.sort_index()
y_test = y_test.sort_index()


print("Training in progress")
clf_svm = svm.SVMkernal(X_train,y_train)
clf_svm.fit(X_train, y_train)

def test():
    predicted = clf_svm.predict(X_test)
    print("Support vector Machine Evaluations :\n")
    print("Testing Accuracy => {}\n".format(clf_svm.score(X_test,y_test) * 100))
    print("Confusion Matrix => \n{}\n".format(confusion_matrix(y_test, predicted)))
    print("Classification Summary => \n{}\n".format(classification_report(y_test, predicted)))
    print("F1 Score => {}\n".format(f1_score(y_test, predicted, average='binary')))
    print("Building Learning Curve")
    plot_learning_curve(clf_svm, "Lerning Curve for polynomial SVM", X_train, y_train, cv=5)
    print("Done! Everything successfully executed")
    

