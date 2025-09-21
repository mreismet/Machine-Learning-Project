import pandas as pd
import json

from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine




class my_class(object):
    with open('trainingdata.json') as peopledata:
        data = json.load(peopledata) # opens the json file as a dataframe
        df = pd.DataFrame(data)

    Mapping = {'low': 0, 'medium': 1, 'high': 2} #contains the mapping to covert riskofdebt into 0, 1 ,2 codes
    
    df['encoded_riskofdebt'] = df['riskofdebt'].map(Mapping) #encodes the riskofdebt into 0 meaning low risk 1 meaning medium risk and 2 meaning high risk

    valuesconsidered = df[['loanAmount','profitToLoanRatio','creditScore','encoded_riskofdebt']] # values considered in the algorithm

    targetvalue = df[['loanApproved']] # result that trains the algorithm

    model = DecisionTreeClassifier(max_depth= 8, random_state=42, min_samples_leaf= 12) #

    model.fit(valuesconsidered,targetvalue) #fits values to train the algorithm
    

    #now that the algorithm is trained its time to put it to the test with 400 files 
    
    file = input("Type 0 if you are working with a excel data file and 1 if you are working with a json file")

    if(file == "1"):
        filename = input("\nType the name of the file and add a .json at the end\n")
    
        with open(filename) as testdata:
            datatest = json.load(testdata) # opens the file and proceeds to load it via a json library
            dftest = pd.DataFrame(datatest)

        dftest['encoded_riskofdebt'] = dftest['riskofdebt'].map(Mapping) #encodes the riskofdebt into 0 meaning low risk 1 meaning medium risk and 2 meaning high risk
        testvalues = dftest[['loanAmount','profitToLoanRatio','creditScore','encoded_riskofdebt']] # obtains all testing values so that it can predict the future value
        targetvaluestest = dftest['loanApproved'] 
    
    
        prediction = model.predict(testvalues)
    
        dftest['loanApproved'] = prediction

        conn_string = 'postgresql://postgres:password@localhost/postgres'
        db = create_engine(conn_string)

        dftest.drop("encoded_riskofdebt",axis = 1,inplace=True) # removes the encoded new value that is used to predict the test data

        dftest.to_sql('datatest', db,if_exists='replace',index = False)
        print("File added to the postgressql database")
    elif(file == "0"):
        filename = input("\nType the name of the file and add a .xlxs at the end\n")

        dftest = pd.read_excel(filename, engine="openpyxl")
        print(dftest)
        
        
        dftest['encoded_riskofdebt'] = dftest['riskofdebt'].map(Mapping)
        testvalues = dftest[['loanAmount','profitToLoanRatio','creditScore','encoded_riskofdebt']]
        targetvaluestest = dftest['loanApproved'] 
    
    
        prediction = model.predict(testvalues)
    
        dftest['loanApproved'] = prediction

        conn_string = 'postgresql://postgres:password@localhost/postgres'
        db = create_engine(conn_string)

        dftest.drop("encoded_riskofdebt",axis = 1,inplace=True) # removes the encoded new value that is used to predict the test data

        dftest.to_sql('datatest', db,if_exists='replace',index = False)

        print("File added to the postgressql database")