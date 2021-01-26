import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

df=pd.read_csv('lungcancer.csv')
X= df.iloc[:,1:].values
y=df.iloc[:,0]



imputer= SimpleImputer(missing_values=np.nan, strategy='median')
imputer=imputer.fit_transform(X)

X= pd.DataFrame(imputer)

conf_matrix_list_of_arrays = []
acc_list_of_arrays=[]
f1_list_of_arrays=[]
kf=KFold(n_splits=5, random_state=7, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    cls = RandomForestClassifier(n_estimators=100,criterion="gini")
    cls.fit(X_train,y_train)
    y_pred=cls.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    acc= metrics.accuracy_score(y_test,y_pred)
    f1=metrics.f1_score(y_test,y_pred, average='micro')
    conf_matrix_list_of_arrays.append(conf_matrix)
    acc_list_of_arrays.append(acc)
    f1_list_of_arrays.append(f1)
print("Random Forest")
print("Confusion Matrix")
print(conf_matrix_list_of_arrays)
print("Accuracy")
print(acc_list_of_arrays)
print("F1 score")
print(f1_list_of_arrays)