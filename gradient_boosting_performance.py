from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import time
import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("Execution time: {:.2f} seconds".format(end_time - start_time))
        return result
    return wrapper


@timeit
def compute_gradient_boosting(project):
    df_train = pd.read_csv(
        f"csv/{project}_fatty_commits_training_without_sp.csv")
    df_test = pd.read_csv(f"csv/{project}_fatty_commits_test_without_sp.csv")

    columns = ['CN', 'TN', 'JC', 'AA', 'PA', 'F-I-F', 'F-A-F', 'F-C-F', 'F-M1-F', 'F-M2-F', 'F-M3-F', 'F-R-F',
               'F-I-D-I-F', 'F-D-I-F', 'F-A-I-F', 'F-I-C-I-F', 'F-I-A-I-F', 'F-I-R-I-F']

    X_train = np.array(df_train[columns])
    y_train = np.array(df_train['CV'])
    X_test = np.array(df_test[columns])
    y_test = np.array(df_test['CV'])

    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    # print(
    #     f"  X_train dim -> {X_train.shape} \n y_train dim -> {y_train.shape}")
    # print(f"  X_test dim -> {X_test.shape} \n y_test dim -> {y_test.shape}")

    gb_clf = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # Train the classifier on the training set
    gb_clf.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = gb_clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)

    print("-"*50)
    print(f"  \tperformance metrics for {project}")
    print("-"*50)
    print(f"Accuracy: {accuracy:.3f}",)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.2f}")
    print(f"AUC-ROC: {auc_roc:.2f}")
    print("-"*50)
    print("\n")
    print(classification_report(y_test, y_pred))


@timeit
def compute_adaboost_metrics(project):
    df_train = pd.read_csv(
        f"csv/{project}_fatty_commits_training_without_sp.csv")
    df_test = pd.read_csv(f"csv/{project}_fatty_commits_test_without_sp.csv")

    columns = ['CN', 'TN', 'JC', 'AA', 'PA', 'F-I-F', 'F-A-F', 'F-C-F', 'F-M1-F', 'F-M2-F', 'F-M3-F', 'F-R-F',
               'F-I-D-I-F', 'F-D-I-F', 'F-A-I-F', 'F-I-C-I-F', 'F-I-A-I-F', 'F-I-R-I-F']

    X_train = np.array(df_train[columns])
    y_train = np.array(df_train['CV'])
    X_test = np.array(df_test[columns])
    y_test = np.array(df_test['CV'])

    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    ada_clf = AdaBoostClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42)

    ada_clf.fit(X_train, y_train)

    y_pred = ada_clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    y_pred_proba = ada_clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    print("-"*50)
    print(f"  \tperformance metrics for {project}")
    print("-"*50)
    print(f"Accuracy: {accuracy:.3f}",)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.2f}")
    print(f"AUC-ROC: {auc_roc:.2f}")
    print("-"*50)
    print("\n")
    print(classification_report(y_test, y_pred))


def write_metrics_report_csv(X_test, y_pred, y_test, classifier, y_pre1, project, fp):

    classifier_name = type(classifier).__name__
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pre1))

    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    # print(f"Accuracy: {accuracy:.6f}",)
    # print(f"Precision: {precision:.6f}")
    # print(f"Recall: {recall:.6f}")
    # print(f"F1-score: {f1:.6f}")
    # print(f"AUC-ROC: {auc_roc:.6f}")
    # print("-"*50)
    # print("\n")
    # print(classification_report(y_test, y_pred))

    return project, classifier_name, accuracy, precision, recall, f1, auc_roc


def run_svm(projects):
    y_pre1 = None

    for project in projects:

        df_train = pd.read_csv(
            f"csv/{project}_fatty_commits_training_without_sp.csv")
        df_test = pd.read_csv(
            f"csv/{project}_fatty_commits_test_without_sp.csv")

        # columns = ['CN', 'TN', 'JC', 'AA', 'PA', 'SPL', 'F-I-F', 'F-A-F', 'F-C-F', 'F-M1-F', 'F-M2-F', 'F-M3-F', 'F-R-F',
        #            'F-I-D-I-F', 'F-D-I-F', 'F-A-I-F', 'F-I-C-I-F', 'F-I-A-I-F', 'F-I-R-I-F']

        columns = ['CN', 'TN', 'JC', 'AA', 'PA', 'F-I-F', 'F-A-F', 'F-C-F', 'F-M1-F', 'F-M2-F', 'F-M3-F', 'F-R-F',
                   'F-I-D-I-F', 'F-D-I-F', 'F-A-I-F', 'F-I-C-I-F', 'F-I-A-I-F', 'F-I-R-I-F']

        X_train = np.array(df_train[columns])
        y_train = np.array(df_train['CV'])
        X_test = np.array(df_test[columns])
        y_test = np.array(df_test['CV'])

        scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
        X_train = scaling.transform(X_train)
        X_test = scaling.transform(X_test)

        print("-"*50)
        print(f"  \tperformance metrics for {project}")
        print("-"*50)

        print("\nSupport Vector Machines (SVM):")
        reg_svc = SVC(kernel='linear')
        reg_svc.fit(X_train, y_train)
        y_pred = reg_svc.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted',
                      labels=np.unique(y_pre1))

        classifier_name = type(classifier).__name__

        y_pred_proba = reg_svc.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        row = [project, classifier_name, accuracy,
               precision, recall, f1, auc_roc]

        with open('performance_metrics.csv', "a") as f:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(row)


def check_string_csv(filename, search_string):
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if search_string in row:
                return True
        return False


def compute_probabilities(projects):

    csv_file = open('issue_performance_metrics.csv', "a")

    for idx, project in enumerate(projects):

        if check_string_csv('issue_performance_metrics.csv', project):
            continue

        rows = []
        print(f"{idx}. {project}")
        df_train = pd.read_csv(
            f"issue_csv/{project}_fatty_commits_training_changed_issue_files_without_sp.csv")
        df_test = pd.read_csv(
            f"issue_csv/{project}_fatty_commits_test_changed_issue_files_without_sp.csv")

        # columns = ['CN', 'TN', 'JC', 'AA', 'PA', 'SPL', 'F-I-F', 'F-A-F', 'F-C-F', 'F-M1-F', 'F-M2-F', 'F-M3-F', 'F-R-F',
        #            'F-I-D-I-F', 'F-D-I-F', 'F-A-I-F', 'F-I-C-I-F', 'F-I-A-I-F', 'F-I-R-I-F']
        # /home/yugandhar/code/summer_internship_2022/issue_csv/bigtop_fatty_commits_test_changed_issue_files_without_sp.csv
        columns = ['CN', 'TN', 'JC', 'AA', 'PA', 'F-I-F', 'F-A-F', 'F-C-F', 'F-M1-F', 'F-M2-F', 'F-M3-F', 'F-R-F',
                   'F-I-D-I-F', 'F-D-I-F', 'F-A-I-F', 'F-I-C-I-F', 'F-I-A-I-F', 'F-I-R-I-F']

        X_train = np.array(df_train[columns])
        y_train = np.array(df_train['CV'])
        X_test = np.array(df_test[columns])
        y_test = np.array(df_test['CV'])

        try:
            oversample = SMOTE()
            X_train, y_train = oversample.fit_resample(X_train, y_train)

        except ValueError:
            print("Value error : found array with 0 samples ")
            continue

        # scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
        # X_train = scaling.transform(X_train)
        # X_test = scaling.transform(X_test)

        # print(X_train.shape)
        # print(y_train.shape)
        # print(X_test.shape)
        # print(y_test.shape)
        print("-"*50)
        print(f"  \tperformance metrics for {project}")
        print("-"*50)
        print("Logistic Regression:")
        reg_log = LogisticRegression(max_iter=10000)
        reg_log.fit(X_train, y_train)
        y_pre1 = reg_log.predict(X_test)
        rows.append(write_metrics_report_csv(
            X_test, y_pre1, y_test, reg_log, y_pre1, project, csv_file))

        print("\nRandom Forest:")
        reg_rf = RandomForestClassifier()
        reg_rf.fit(X_train, y_train)
        y_pre2 = reg_rf.predict(X_test)
        rows.append(write_metrics_report_csv(
            X_test, y_pre2, y_test, reg_rf, y_pre1, project, csv_file))

        print("\nDecision Tree:")
        reg_dec = DecisionTreeClassifier()
        reg_dec.fit(X_train, y_train)
        y_pre4 = reg_dec.predict(X_test)
        rows.append(write_metrics_report_csv(
            X_test, y_pre4, y_test, reg_dec, y_pre1, project, csv_file))

        print("\nNaive Bayes :")
        reg_bay = GaussianNB()
        reg_bay.fit(X_train, y_train)
        y_pre5 = reg_bay.predict(X_test)
        rows.append(write_metrics_report_csv(
            X_test, y_pre5, y_test, reg_bay, y_pre1, project, csv_file))

        print("\nAda Boost Classifier")
        ada_clf = AdaBoostClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42)

        ada_clf.fit(X_train, y_train)

        y_pred_adaboost = ada_clf.predict(X_test)
        y_pred_adaboost = ada_clf.predict(X_test)
        rows.append(write_metrics_report_csv(X_test, y_pred_adaboost, y_test,
                                             ada_clf, y_pre1, project, csv_file))

        print(f"\nGradient Boosting Classifier ")
        gb_clf = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

        gb_clf.fit(X_train, y_train)

        y_pred_gradient_boost = gb_clf.predict(X_test)
        rows.append(write_metrics_report_csv(X_test, y_pred_gradient_boost,
                                             y_test, gb_clf, y_pre1, project, csv_file))

        # print("\nSupport Vector Machines (SVM):")
        # reg_svc = SVC(kernel='linear', probability=True)
        # reg_svc.fit(X_train, y_train)
        # y_pred_svm = reg_svc.predict(X_test)
        # rows.append(write_metrics_report_csv(X_test, y_pred_svm,
        #                                      y_test, reg_svc, y_pre1, project, csv_file))

        csv_writer = csv.writer(csv_file)
        for row in rows:
            csv_writer.writerow(row)

        # print("------------------------------------------------------------------------------------")

    csv_file.close()

    print("------------------------------------------------------------------------------------")


def main():

    projects = ['mahout', 'pdfbox', 'opennlp', 'openwebbeans', 'mina-sshd', 'helix', 'curator',
                'storm', 'cxf-fediz',
                'knox', 'zeppelin', 'samza',
                'directory-kerby', 'pig', 'manifoldcf',
                'giraph', 'bigtop', 'kafka', 'oozie',
                'falcon', 'deltaspike', 'calcite',
                'parquet-mr', 'tez', 'lens', 'phoenix',
                'kylin', 'ranger']

    #print(check_string_csv('issue_performance_metrics.csv', 'mahout'))

    # run_svm(projects[::-1])

    # for project in projects[::-1]:
    #     print(project)
    # user_input = input(
    #     f"  Do you want to continue with {project}? (y/n): ").lower()

    # if user_input == "y":
    #     pass
    # else:
    #     break

    # compute_gradient_boosting(project)
    compute_probabilities(projects)
    # compute_adaboost_metrics(project)

    print(" -------- Finished running --------")


if __name__ == "__main__":
    main()
