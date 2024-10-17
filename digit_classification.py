
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score

# Load the dataset
digits = load_digits()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

# Logistic Regression
lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(x_train, y_train)
print("Logistic Regression Score:", lr.score(x_test, y_test))

# SVC
svc = SVC()
svc.fit(x_train, y_train)
print("SVC Score:", svc.score(x_test, y_test))

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=40)
rf.fit(x_train, y_train)
print("Random Forest Score:", rf.score(x_test, y_test))

# KFold
kf = KFold(n_splits=3)
for train_index, test_index in kf.split([1, 2, 3, 4, 5, 6, 7, 8, 9]):
    print("Train:", train_index, "Test:", test_index)

# Define a function to calculate model score
def get_score(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

print("SVC KFold Score:", get_score(SVC(), x_train, x_test, y_train, y_test))
print("Random Forest KFold Score:", get_score(RandomForestClassifier(n_estimators=40), x_train, x_test, y_train, y_test))
print("Logistic Regression KFold Score:", get_score(LogisticRegression(), x_train, x_test, y_train, y_test))

# StratifiedKFold
folds = StratifiedKFold(n_splits=3)
score_logistic, score_SVM, score_rf = [], [], []

for train_index, test_index in folds.split(digits.data, digits.target):
    x_train, x_test = digits.data[train_index], digits.data[test_index]
    y_train, y_test = digits.target[train_index], digits.target[test_index]

    score_logistic.append(get_score(LogisticRegression(), x_train, x_test, y_train, y_test))
    score_SVM.append(get_score(SVC(), x_train, x_test, y_train, y_test))
    score_rf.append(get_score(RandomForestClassifier(n_estimators=40), x_train, x_test, y_train, y_test))

print("Logistic Regression StratifiedKFold Scores:", score_logistic)
print("SVC StratifiedKFold Scores:", score_SVM)
print("Random Forest StratifiedKFold Scores:", score_rf)

# Cross-validation
print("Cross-validation Logistic Regression:", cross_val_score(LogisticRegression(), digits.data, digits.target, cv=5))
print("Cross-validation Random Forest:", cross_val_score(RandomForestClassifier(), digits.data, digits.target, cv=5))

# Average cross-validation scores for Random Forest with different estimators
scores1 = cross_val_score(RandomForestClassifier(n_estimators=5), digits.data, digits.target, cv=10)
print("Random Forest (5 estimators) Average Score:", np.average(scores1))

scores1 = cross_val_score(RandomForestClassifier(n_estimators=20), digits.data, digits.target, cv=20)
print("Random Forest (20 estimators) Average Score:", np.average(scores1))

scores1 = cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target, cv=30)
print("Random Forest (40 estimators) Average Score:", np.average(scores1))
