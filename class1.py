import pandas as pd
import json
import sklearn as sk
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree


class my_class(object):
    with open('trainingdata.json') as peopledata:
        data = json.load(peopledata)
        df = pd.DataFrame(data)

    valuesconsidered = df[['loanAmount','profitToLoanRatio','creditScore']]

    print(df)

    targetvalue = df[['loanApproved']]

    

    valuesconsidered_train, valuesconsidered_test, targetvalue_train, targetvalue_test = train_test_split(valuesconsidered,targetvalue, test_size=0.2,random_state=42)

    model = DecisionTreeClassifier(max_depth= 10, random_state=42, min_samples_leaf= 12)

    model.fit(valuesconsidered_train,targetvalue_train)

    predictions = model.predict(valuesconsidered_test)


    print(predictions)
   
    print(targetvalue_test)
    
    print("accuracy", metrics.accuracy_score(predictions,targetvalue_test))


    with open('testingdata.json') as testdata:
        datatest = json.load(testdata)
        dftest = pd.DataFrame(datatest)


    testvalues = dftest[['loanAmount','profitToLoanRatio','creditScore']]

    targetvaluetest = dftest[['loanApproved']]
    

    prediction = model.predict(testvalues)

    print("accuracy", metrics.accuracy_score(prediction,targetvaluetest))

    tree.plot_tree(model, proportion=True)
    plt.show()
    








