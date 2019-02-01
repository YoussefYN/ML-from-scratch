import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('titanic.csv')
data = data.drop(columns=['name'])
data = data.fillna(data.mean())
data = pd.get_dummies(data, columns=['sex','embarked'])

X = data.iloc[:,1:].astype('float64').values
y = data.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

#TO DO ---- 10 POINTS --------------------- Average accuracy --------------------
# Caclulate average accuracy for estimators in the ensemble
# Solution ----------------------------------------------------------
accuracies = []
for estimators in model.estimators_:
    y_pred_est = estimators.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_est)
    accuracies.append(accuracy)

print(np.mean(accuracies))
# ----------------------------------------------------------


#TO DO ---- 10 POINTS --------------------- Predictor --------------------
# Implement prediction function for the RandomForestClassifier
def predict(estimators, X_test):
    pass
# Solution ----------------------------------------------------------
def predict(estimators, X_test):
    y_est_preds = []
    y_pred = []
    for m in estimators:
        y_est_preds.append(m.predict(X_test))
    
    counts = []
    for s in range(X_test.shape[0]):
        count = 0
        for i in range(len(estimators)):
            count += y_est_preds[i][s]
        counts.append(count)
        if count < len(estimators) / 2:
            y_pred.append(0)
        else:
            y_pred.append(1)  
    
    return y_pred
# ----------------------------------------------------------
y_my_pred = predict(model.estimators_, X_test)
print(accuracy_score(y_pred, y_my_pred))
