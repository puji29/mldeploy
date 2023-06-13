import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib

df = pd.read_csv("D:/SKRIPSI/fitur glcm 224.csv")

X = df.iloc[:,0:4]
y = df.iloc[:,4]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

gnb = GaussianNB()
clf = gnb.fit(X_train, y_train)


joblib.dump(clf, "clf.pkl")