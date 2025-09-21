import pandas as pd
import json
import sklearn as sk

class my_class(object):
    with open('peopledata.json') as peopledata:
        data = json.load(peopledata)
        df = pd.DataFrame(data)

    from sk. import decision_tree
    print(df)









