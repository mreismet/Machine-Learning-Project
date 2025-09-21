import pandas as pd
import json
import sklearn as sk
import numpy as np
import os
import psycopg2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sqlalchemy import create_engine
from sklearn.preprocessing import OneHotEncoder


class my_class(object):
    with open('trainingdata.json') as peopledata:
        data = json.load(peopledata)
        df = pd.DataFrame(data)
    


    Maping = {'low': 0, 'medium': 1, 'high': 2}
    
    df['encoded_riskofdebt'] = df['riskofdebt'].map(Maping)

    print(df['encoded_riskofdebt'])

    valuesconsidered = df[['loanAmount','profitToLoanRatio','creditScore','encoded_riskofdebt']]
    print(df)
    targetvalue = df[['loanApproved']]

    valuesconsidered_train, valuesconsidered_test, targetvalue_train, targetvalue_test = train_test_split(valuesconsidered,targetvalue, test_size=0.2,random_state=42)
    
    model = DecisionTreeClassifier(max_depth= 8, random_state=42, min_samples_leaf= 12)

    model.fit(valuesconsidered_train,targetvalue_train) #fits new values to train
    

    predictions = model.predict(valuesconsidered_test)
    print("accuracy", metrics.accuracy_score(predictions,targetvalue_test))
    
    
    filename = 'file.xlsx'
    dftest = pd.read_excel(filename, engine="openpyxl")
        

    dftest['encoded_riskofdebt'] = dftest['riskofdebt'].map(Maping)
    testvalues = dftest[['loanAmount','profitToLoanRatio','creditScore','encoded_riskofdebt']]
    targetvaluestest = dftest['loanApproved'] 
    
    
    prediction = model.predict(testvalues)
    

    print("accuracy", metrics.accuracy_score(prediction,targetvaluestest))
    

    conn_string = 'postgresql://postgres:password@localhost/postgres'
    db = create_engine(conn_string)

    dftest.drop("encoded_riskofdebt",axis = 1,inplace=True) # removes the encoded new value that is used to predict the test data

    dftest.to_sql('datatest', db,if_exists='replace',index = False)


   
    








