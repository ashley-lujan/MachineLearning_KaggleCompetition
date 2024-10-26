import numpy as np
from sklearn.ensemble import RandomForestClassifier
from main import process, save_predictions, saveDataSet, saveTestSet, saveXY
from sklearn.model_selection import train_test_split

def linearClassifier(): 
    X, y = saveXY('train_final.csv')
    X = X[1:]
    X = process(X)
    
    y = y[1:]
    print(y[:5])

    print("learning")
    # clf = RandomForestClassifier(max_depth=1, random_state=14)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Create a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    print("predicting")
    testX = saveTestSet('test_final.csv')[1:]
    testX = process(testX)
    save_predictions(clf.predict, testX, "submissionRandomForest.csv")
    return 0

if __name__ == "__main__":
    linearClassifier()