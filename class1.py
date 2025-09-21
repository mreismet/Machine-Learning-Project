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
    with open('peopledata.json') as peopledata:
        data = json.load(peopledata)
        df = pd.DataFrame(data)


    valuesconsidered = df[['loan_amount','profit_to_loan_ratio','credit_score']]

    targetvalue = df[['loan_approved']]

    valuesconsidered_train, valuesconsidered_test, targetvalue_train, targetvalue_test = train_test_split(valuesconsidered,targetvalue, test_size=0.2,random_state=42)

    model = DecisionTreeClassifier(max_depth= 3, random_state=42)

    model.fit(valuesconsidered_train,targetvalue_train)

    predictions = model.predict(valuesconsidered_test)


    print(predictions)
   
    print(targetvalue_test)
    
    print("accuracy", metrics.accuracy_score(predictions,targetvalue_test))

    tree.plot_tree(model, proportion=True)
    plt.show()








