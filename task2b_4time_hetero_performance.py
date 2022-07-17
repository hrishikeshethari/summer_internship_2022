import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from collections import Counter


def cross_validation(model, _X, _y, _cv):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                             X=_X,
                             y=_y,
                             cv=_cv,
                             scoring=_scoring,
                             return_train_score=True)
    return {"Training Accuracy scores": results['train_accuracy'],
            "Mean Training Accuracy": results['train_accuracy'].mean() * 100,
            "Training Precision scores": results['train_precision'],
            "Mean Training Precision": results['train_precision'].mean(),
            "Training Recall scores": results['train_recall'],
            "Mean Training Recall": results['train_recall'].mean(),
            "Training F1 scores": results['train_f1'],
            "Mean Training F1 Score": results['train_f1'].mean(),
            "Validation Accuracy scores": results['test_accuracy'],
            "Mean Validation Accuracy": results['test_accuracy'].mean() * 100,
            "Validation Precision scores": results['test_precision'],
            "Mean Validation Precision": results['test_precision'].mean(),
            "Validation Recall scores": results['test_recall'],
            "Mean Validation Recall": results['test_recall'].mean(),
            "Validation F1 scores": results['test_f1'],
            "Mean Validation F1 Score": results['test_f1'].mean()
            }


# df_train = pd.read_csv("task2b_training_imbalance.csv")
# df_test = pd.read_csv("task2b_test_imbalance.csv")

# df_train = pd.read_csv("task2b_training_no_imbalance.csv")
# df_test = pd.read_csv("task2b_test_no_imbalance.csv")

# df_train = pd.read_csv("task2b_training_imbalance_6time.csv")
# df_test = pd.read_csv("task2b_test_imbalance_6time.csv")

# df_train = pd.read_csv("task2b_training_no_imbalance_6time.csv")
# df_test = pd.read_csv("task2b_test_no_imbalance_6time.csv")

df_train = pd.read_csv("task2b_training_imbalance_6time_t1.csv")
df_test = pd.read_csv("task2b_test_imbalance_6time_t1.csv")

# df_train = pd.read_csv("task2b_training_no_imbalance_6time_t1.csv")
# df_test = pd.read_csv("task2b_test_no_imbalance_6time_t1.csv")

columns = ['CN', 'TN', 'JC', 'AA', 'PA', 'SPL', 'F-I-F', 'F-A-F', 'F-C-F', 'F-M1-F', 'F-M2-F', 'F-M3-F', 'F-R-F',
           'F-I-D-I-F', 'F-D-I-F', 'F-A-I-F', 'F-I-C-I-F', 'F-I-A-I-F', 'F-I-R-I-F']
# columns = ['CN', 'TN', 'JC', 'AA', 'PA', 'SPL']
X_train = np.array(df_train[columns])
y_train = np.array(df_train['CV'])
X_test = np.array(df_test[columns])
y_test = np.array(df_test['CV'])

# oversample = SMOTE()
# X_train, y_train = oversample.fit_resample(X_train, y_train)
# X_test, y_test = oversample.fit_resample(X_test, y_test)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print("Logistic Regression:")
reg_log = LogisticRegression(max_iter=10000)
'''r1 = cross_validation(reg_log, X, y, 10)
print(r1)
mean_score1 = cross_val_score(reg_log, X, y, scoring="roc_auc", cv=10).mean()
print(f"roc_auc_score: {mean_score1}")'''
# print("-------------------------------------------")
reg_log.fit(X_train, y_train)
y_pre1 = reg_log.predict(X_test)
# print(metrics.classification_report(y_test, y_pre1))
print("roc_auc_score: ", roc_auc_score(y_test, y_pre1))
# print("f1 score: ", f1_score(y_test, y_pre1, average='weighted', labels=np.unique(y_pre1)))

print("\nRandom Forest:")
reg_rf = RandomForestClassifier()
'''r2 = cross_validation(reg_rf, X, y, 10)
print(r2)
mean_score2 = cross_val_score(reg_rf, X, y, scoring="roc_auc", cv=10).mean()
print(f"roc_auc_score: {mean_score2}")'''
# print("-------------------------------------------")
reg_rf.fit(X_train, y_train)
y_pre2 = reg_rf.predict(X_test)
# print(metrics.classification_report(y_test, y_pre2))
print("roc_auc_score: ", roc_auc_score(y_test, y_pre2))
# print("f1 score: ", f1_score(y_test, y_pre2, average='weighted', labels=np.unique(y_pre2)))
# columns_df = pd.DataFrame({'Importance': reg_rf.feature_importances_, 'Features': columns})
# print(columns_df)

print("\nDecision Tree:")
reg_dec = DecisionTreeClassifier()
'''r4 = cross_validation(reg_dec, X, y, 10)
print(r4)
mean_score4 = cross_val_score(reg_dec, X, y, scoring="roc_auc", cv=10).mean()
print(f"roc_auc_score: {mean_score4}")'''
# print("-------------------------------------------")
reg_dec.fit(X_train, y_train)
y_pre4 = reg_dec.predict(X_test)
# print(metrics.classification_report(y_test, y_pre4))
print("roc_auc_score: ", roc_auc_score(y_test, y_pre4))
# print("f1 score: ", f1_score(y_test, y_pre4, average='weighted', labels=np.unique(y_pre4)))

print("\nNaive Bayes :")
reg_bay = GaussianNB()
'''r5 = cross_validation(reg_bay, X, y, 10)
print(r5)
mean_score5 = cross_val_score(reg_bay, X, y, scoring="roc_auc", cv=10).mean()
print(f"roc_auc_score: {mean_score5}")'''
# print("-------------------------------------------")
reg_bay.fit(X_train, y_train)
y_pre5 = reg_bay.predict(X_test)
# print(metrics.classification_report(y_test, y_pre5))
print("roc_auc_score: ", roc_auc_score(y_test, y_pre5))
# print("f1 score: ", f1_score(y_test, y_pre5, average='weighted', labels=np.unique(y_pre5)))

'''print("\nSupport Vector Machines (SVM):")
reg_svc = SVC(kernel='linear')
r3 = cross_validation(reg_svc, X, y, 10)
print(r3)
mean_score3 = cross_val_score(reg_svc, X, y, scoring="roc_auc", cv=10).mean()
print(f"roc_auc_score: {mean_score3}")
print("-------------------------------------------")
reg_svc.fit(X_train, y_train)
y_pre3 = reg_svc.predict(X_test)
print(metrics.classification_report(y_test, y_pre3))
print("roc_auc_score: ", roc_auc_score(y_test, y_pre3))
print("f1 score: ", f1_score(y_test, y_pre3))'''
