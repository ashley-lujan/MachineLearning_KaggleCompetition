import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from main import process, save_predictions, saveDataSet, saveTestSet, saveXY

def linearClassifier(): 
    X, y = saveXY('train_final.csv')
    X = X[1:]
    X = process(X)
    
    y = y[1:]
    print(y[:5])

    print("learning")
    clf = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=1000, tol=1e-3))
    clf.fit(X, y)

    print("predicting")
    testX = saveTestSet('test_final.csv')[1:]
    testX = process(testX)
    save_predictions(clf.predict, testX, "submissionClassifer.csv")
    return 0

if __name__ == "__main__":
    linearClassifier()