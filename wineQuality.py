#Importing the libraries


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

#Importing Dataset
data=pd.read_excel('d:/corizo/Major project/Red_wine.xlsx')

"""
Missing value
 
"""

#finding and replacing the missing value into the required data column

data['quality'] = np.where(data['quality'].isna(), data['quality'].fillna(data['quality'].mean()),
                                              data['quality'])
print("How many Alcohol or quality data are missing : ",data['alcohol'].isna().sum(),data['quality'].isna().sum())

"""
Data analysis

#1

We will compare all the other feature directly propotional or not with the target 'Quality'
feature Selection Using plotting Correlatin Matrix and will choose the feature which is 
highly correlated with 'qualitty'
1.Poistive Correlation
2.Negative Correlation

"""
co_mat=data.corr() #return the correlational matrix
plt.figure(figsize=(15,15))
chart=sns.heatmap(co_mat, fmt='.1f',annot=True,cmap='Blues')
plt.show()

"""
#2
Data Preprocessing
    -X
    -Y label binarisation
    1 (good quality) if q > = 7
    0 (bad quality ) if q < 7

StandardScaler: StandardScaler is used to resize the distribution of values
        ​​so that the mean of the observed values ​​is 0 and the standard deviation
        is 1.
"""

X=data.iloc[:,0:11]
y=data['quality'].apply(lambda q: 1 if q>=7 else 0)

"""
Splitting Dataset into
    -training set
    -test set

"""

X=np.asarray(X.alcohol)
y=np.asarray(y)

dSet=np.random.permutation(np.arange(y.size))

nTrain=int(X.size * .85)
nTest=dSet.size-nTrain

#Training Set
XTrain=X[dSet[:nTrain]]
YTrain=y[dSet[:nTrain]]

#Test Set
XTest=X[dSet[nTrain:]]
YTest=y[dSet[nTrain:]]

#Re-distributing Data
sc=StandardScaler()
XTrain=sc.fit_transform(XTrain[:,np.newaxis])
XTest=sc.fit_transform(XTest[:,np.newaxis])
print(XTrain)
print(YTrain)

"""
#3

Training Model: Support Vector Classifier

"""


model = SVC(C = 1.4, gamma = 0.1, kernel = 'rbf')
model.fit(XTrain,YTrain)
y_pred = model.predict(XTest)

"""
#4

Model Evaluation
    -Accuracy on test data 

"""
print(classification_report(YTest, y_pred))
print("training accuracy :", model.score(XTrain, YTrain))
print("testing accuracy :", model.score(XTest, YTest))
