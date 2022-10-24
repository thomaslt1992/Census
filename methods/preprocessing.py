from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
import numpy as np
import pandas as pd

def encode_country(country):

    high_income = [' United-States',' Englang', ' Germany', ' Iran', ' Italy', ' Poland', ' Portugal',
    ' France',' China',' Japan', ' Scotland' ,' Greece',' Hong',' Ireland',' Hungary',' Holand-Netherlands']
    if country in high_income:
        return 1
    else:
        return 0


def encode_ordinal_categories(X,categories):

    for cat in categories:
        mlist = list(np.unique(X[cat]))
        ord_enc = OrdinalEncoder(categories=[mlist])
        new_col = cat + '_encoded'
        X[new_col] = ord_enc.fit_transform(X.loc[:,[cat]])
        X = X.drop(columns=cat)
        return X


def encode_one_hot_categories(X,categories):

    for cat in categories:
        one_hot = OneHotEncoder(sparse=False)
        one_hot.fit(np.asarray(X[cat]).reshape(-1,1))
        df = one_hot.transform(np.asarray(X[cat]).reshape(-1,1))
        df = pd.DataFrame(data=df, columns = one_hot.categories_)
        X = X.drop(columns=[cat])
        X = pd.concat([X,df],axis=1)
    
    return X


def encode_binary_categories(X,categories):
    le = LabelEncoder()

    for cat in categories:
        X[cat] = le.fit_transform(X[cat])

    return X
    
