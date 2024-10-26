import numpy as np
from sklearn.ensemble import RandomForestClassifier
from main import process, save_predictions, saveDataSet, saveTestSet, saveXY

def linearClassifier(): 
    X, y = saveXY('train_final.csv')
    X = X[1:]
    X = process(X)
    
    y = y[1:]
    print(y[:5])

    print("learning")
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)

    print("predicting")
    testX = saveTestSet('test_final.csv')[1:]
    testX = process(testX)
    save_predictions(clf.predict, testX, "submissionClassifer.csv")
    return 0

if __name__ == "__main__":
    linearClassifier()