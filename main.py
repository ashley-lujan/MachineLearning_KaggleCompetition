import pandas as pd
from decAlgo import DecisionTree


def linear_classification(): 
    data = pd.read_csv('train_final.csv')
    
def saveDataSet(filename):
    result = []
    inti = 0
    # with open(filename + '/train.csv', 'r') as f:
    with open(filename, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            inti += 1
            result.append(terms)
    return result

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
    

def save_preidictions(d3, testing_set): 
    f = open("submission1.csv", "a")
    f.write("ID,Prediction\n")
    for i, test in enumerate(testing_set):
        f.write(str(i + 1) + "," + str(d3.predict(test)) + "\n")
    
    f.close()


if __name__ == "__main__":
    single_decision_tree()
    # linear_classification()
    # print("Hello, World!")