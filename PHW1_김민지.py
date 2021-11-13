import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('breast-cancer-wisconsin.data',na_values = ['?'])
print(df.head(20))
column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
               'Uniformity of Cell Shape', 'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei0','Bland Chromatin',"Normal Nucleoli","Mitoses","Class"]
data = df.values.tolist()
df = pd.DataFrame(data, columns=column_name)

print(df.info())
df.dropna(axis=0,inplace= True)
print(df)
print(df.info())


def ordinalEncode_category(df, str):
    ordinalEncoder = preprocessing.OrdinalEncoder()
    X = pd.DataFrame(df[str])
    ordinalEncoder.fit(X)
    df[str] = pd.DataFrame(ordinalEncoder.transform(X))
# def ordinalEncode_category(df, col_list):
#     ordinalEncoder = preprocessing.OrdinalEncoder()
#     for col in col_list:
#         X = pd.DataFrame(df[col])
#         ordinalEncoder.fit(X)
#         df[col] = pd.DataFrame(ordinalEncoder.transform(X))

def oneHotEncode_category(arr,str):
    enc=preprocessing.OneHotEncoder()
    encodedData=enc.fit_transform(arr[[str]])
    encodedDataRecovery=np.argmax(encodedData,axis=1).reshape(-1,1)
    arr[str] = encodedDataRecovery
# def oneHotEncode_category(arr,col_list):
#     enc=preprocessing.OneHotEncoder()
#     for col in col_list:
#         encodedData = enc.fit_transform(arr[[col]])
#         encodedDataRecovery = np.argmax(encodedData, axis=1).reshape(-1, 1)
#         arr[col] = encodedDataRecovery

def scalingData(dataset, scaled_col):
    scaling=[]
    for encoder in [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]:
        new_df = dataset.copy();
        for col in scaled_col:
            new_df[col] = encoder.fit_transform(new_df[col].values[:,np.newaxis]).reshape(-1)
        scaling.append(new_df)
    return scaling



def encodingNscalingData(dataset, scaled_col, encoded_col):
    result=[]
    for encoder in [0, 1]:
        new_df = dataset.copy();
        if encoder == 0:
            ordinalEncode_category(new_df, encoded_col)
        elif encoder == 1:
            oneHotEncode_category(new_df, "Sample code number")
        new_df.dropna(axis=0, inplace=True)
        for scaler in [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]:
            for col in scaled_col:
                new_df[col] = scaler.fit_transform(new_df[col].values[:, np.newaxis]).reshape(-1)
            result.append(new_df)
    return result

def scalingNencodingData(dataset, scaled_col, encoded_col):
    result=[]
    for scaler in [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]:
        new_df = dataset.copy();
        for col in scaled_col:
            new_df[col] = scaler.fit_transform(new_df[col].values[:,np.newaxis]).reshape(-1)
        new_df.dropna(axis=0, inplace=True)
        for encoder in [0, 1]:
            if encoder == 0:
                ordinalEncode_category(new_df, encoded_col)
            elif encoder == 1:
                oneHotEncode_category(new_df, encoded_col)
            new_df.dropna(axis=0, inplace=True)
            new_df[encoded_col] = scaler.fit_transform(new_df[encoded_col].values[:, np.newaxis]).reshape(-1)
            new_df.dropna(axis=0, inplace=True)
            result.append(new_df)
    return result


encodingNscaling=encodingNscalingData(df,['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
               'Uniformity of Cell Shape', 'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei0','Bland Chromatin',"Normal Nucleoli","Mitoses"],'Sample code number')
print(encodingNscaling)

scalingNencoding=scalingNencodingData(df,['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
               'Uniformity of Cell Shape', 'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei0','Bland Chromatin',"Normal Nucleoli","Mitoses"],'Sample code number')
print(scalingNencoding)


def callParameters(num):
    decisionTreepParameters = {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [3, 5, 10],
        "min_samples_leaf": [1, 2, 3],
        "min_samples_split": [3, 5, 2],
        "max_features": ["auto", "sqrt", "log2"]
    }
    logisticRegressionParameters = {
        "penalty":['l2','l1'],
        "solver":['saga',"liblinear"],
        "multi_class": ['auto', 'ovr'],
        "random_state": [3, 5, 10],
        "C": [1.0,0.5],
        "max_iter" : [1000]
    }
    svmParameters={
        "decision_function_shape": ['ovo', 'ovr'],
        "gamma": ['scale', 'auto'],
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        "C": [1.0, 0.5]
    }
    if num==0:
        return logisticRegressionParameters
    elif num==1:
        return decisionTreepParameters
    else: return svmParameters


def findBestModel (data_list):
    bestscore=-1
    i=0
    for dataset in data_list:
        i=0
        y = dataset["Class"]
        x=dataset.drop(["Class"],axis=1)
        train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.7, random_state=0)
        for model in [LogisticRegression(),DecisionTreeClassifier(), svm.SVC()]:
            tunedModel = GridSearchCV(model, callParameters(i), scoring='neg_mean_squared_error', cv=5)
            tunedModel.fit(train_x, train_y)
            print(tunedModel.best_params_)
            print(tunedModel.best_score_)
            i = i + 1
            if bestscore<tunedModel.best_score_:
                bestscore=tunedModel.best_score_
                bestparams=tunedModel.best_params_
    best=[]
    best.append(bestparams)
    best.append(bestscore)
    return best

print(findBestModel(encodingNscaling))
print(findBestModel(scalingNencoding))




