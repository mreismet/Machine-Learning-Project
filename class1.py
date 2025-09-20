import pandas as pd;
import json;

class my_class(object):
    with open('peopledata.json') as peopledata:
        data = json.load(peopledata)
        df = pd.DataFrame(data)



    print(df)





