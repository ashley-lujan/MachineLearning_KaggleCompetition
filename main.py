import pandas as pd
from decAlgo import DecisionTree
from random import randint
from sklearn.ensemble import AdaBoostClassifier

def linear_classification(): 
    data = pd.read_csv('train_final.csv')
    
def saveDataSet(filename):
    result = []
    inti = 0
    # with open(filename + '/train.csv', 'r') as f:
    with open(filename, 'r') as f:
        for line in f[1:]:
            terms = line.strip().split(',')
            inti += 1
            result.append(terms)
    return result

def saveXY(filename):
    X = []
    y = []
    with open(filename, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            X.append(terms[:-1])
            y.append(terms[-1])
    return (X, y)

def saveTestSet(filename):
    result = []
    inti = 0
    # with open(filename + '/train.csv', 'r') as f:
    with open(filename, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            inti += 1
            result.append(terms[1:])
    return result

def getAttributes(): 
    data = {
    "age": "numeric",
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked", "?"],
    "fnlwgt": "numeric",
    "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool", "?"],
    "education-num": "numeric",
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse", "?"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces", "?"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried", "?"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black", "?"],
    "sex": ["Female", "Male", "?"],
    "capital-gain": "numeric",
    "capital-loss": "numeric",
    "hours-per-week": "numeric",
    "native-country": [
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", 
        "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", 
        "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", 
        "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands", "?"
    ]
    }
    return data



def single_decision_tree(): 
    training_data = saveDataSet('train_final.csv')[1:]
    d3 = DecisionTree(dataSet=training_data, attributesWithValues=getAttributes(), hasNumerics=True, max_depth=12, replaceMissing=True, measurement="Ent")

    testing_set = saveTestSet('test_final.csv')[1:]
    save_preidictions(d3, testing_set)

def randomized_set(l, size): 
    result = []
    for i in range(size): 
        random_index = randint(0, size-1)
        result.append(l[random_index])
    
    return result


def boosting_and_bagging(): 
    training_data = saveDataSet('train_final.csv')[1:]
    trees = []
    T = 30
    max_depth = 3
    training_size = 200
    for i, tree in enumerate(range(T)): 
        random_data = randomized_set(training_data, training_size)
        d3 = DecisionTree(dataSet=training_data, attributesWithValues=getAttributes(), hasNumerics=True, max_depth=3, replaceMissing=True, measurement="Ent")
        trees.append(d3)
        if i % 5 == 0: 
            print("calculating tree", i)

        

    testing_set = saveTestSet('test_final.csv')[1:]
    save_bagging_predictions(trees, testing_set)
    # save_preidictions(d3, testing_set)

def save_bagging_predictions(trees, testing_set): 
    f = open("submission2.csv", "a")
    f.write("ID,Prediction\n")
    for i, test in enumerate(testing_set):
        f.write(str(i + 1) + "," + str(final_prediction(trees, test)) + "\n")
    
    f.close() 

def final_prediction(trees, test):
    sum = 0
    for tree in trees: 
        pred = tree.predict(test)
        sum += (int(pred) - 1)
    if sum > 0: 
        return 1
    return 0 

def process(_X): 
    X = []
    attributes = getAttributes()
    for index in range(len(_X)): 
        _Xi = _X[index]
        Xi = []
        for j, key in enumerate(attributes.keys()): 
            _Xij = _Xi[j]
            if type(attributes[key]) == list:                
                Xi.append(attributes[key].index(_Xij))
            else:
                Xi.append(_Xij)
        X.append(Xi)
    return X






def adaBoost(): 
    X, y = saveXY('train_final.csv')
    X = X[1:]
    X = process(X)
    
    y = y[1:]
    print(y[:5])

    print("learning")
    clf = AdaBoostClassifier(n_estimators = 1000, learning_rate=10, random_state = 10)
    clf.fit(X, y)

    print("predicting")
    testX = saveTestSet('test_final.csv')[1:]
    testX = process(testX)
    save_predictions(clf.predict, testX, "submissionAda.csv")

def save_predictions(predictor_func, test_set, output_file): 
    with open(output_file, "w") as f: 
        f.write("ID,Prediction\n")
        for i, test in enumerate(test_set):
            f.write(str(i + 1) + "," + str(predictor_func([test])[0]) + "\n")



def save_preidictions(d3, testing_set): 
    with open("submission1.csv", "w") as f:
        f.write("ID,Prediction\n")
        for i, test in enumerate(testing_set):
            f.write(str(i + 1) + "," + str(d3.predict(test)) + "\n")
    


if __name__ == "__main__":
    # single_decision_tree()
    # boosting_and_bagging()
    # linear_classification()
    # print("Hello, World!")
    adaBoost()