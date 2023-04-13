import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV

# p = ['xerces2-j',
#      'xmlgraphics-batik',
#      'commons-beanutils',
#      'commons-collections',
#      'commons-dbcp',
#      'commons-digester',
#      'jspwiki',
#      'santuario-java',
#      'commons-bcel',
#      'commons-validator',
#      'commons-io',
#      'commons-jcs',
#      'commons-jexl',
#      'commons-vfs',
#      'commons-lang']

# p = ['xerces2-j',
#      'xmlgraphics-batik',
#      'commons-beanutils',
#      'commons-collections',
#      'commons-dbcp',
#      'commons-digester',
#      'jspwiki',
#      'santuario-java',
#      'commons-bcel',
#      'commons-validator',
#      'commons-io',
#      'commons-jcs',
#      'commons-jexl',
#      'commons-vfs',
#      'commons-lang',
#      'jena',
#      'commons-codec',
#      'commons-math',
#      'maven',
#      'commons-compress',
#      'commons-configuration',
#      'wss4j',
#      'derby',
#      'jackrabbit',
#      'nutch',
#      'httpcomponents-core',
#      'roller',
#      'ant-ivy',
#      'commons-scxml',
#      'archiva',
#      'activemq',
#      'httpcomponents-client',
#      'struts',
#      'openjpa',
#      'directory-studio',
#      'cayenne',
#      'tika',
#      'commons-imaging',
#      'zookeeper',
#      'mahout',
#      'pdfbox',
#      'opennlp',
#      'openwebbeans',
#      'mina-sshd',
#      'pig',
#      'manifoldcf',
#      'freemarker',
#      'gora',
#      'giraph',
#      'helix',
#      'curator',
#      'bigtop',
#      'kafka',
#      'flume',
#      'oozie',
#      'directory-fortress-core',
#      'storm',
#      'falcon',
#      'cxf-fediz',
#      'deltaspike',
#      'systemml',
#      'calcite',
#      'fineract',
#      'parquet-mr',
#      'knox',
#      'streams',
#      'tez',
#      'lens',
#      'zeppelin',
#      'samza',
#      'phoenix',
#      'directory-kerby',
#      'kylin',
#      'commons-rdf',
#      'ranger',
#      'nifi',
#      'eagle'
#      ]

# p = ['mahout', 'pdfbox', 'pig', 'manifoldcf',
#      'gora', 'giraph', 'helix', 'curator', 'bigtop', 'kafka', 'oozie',
#      'directory-fortress-core', 'falcon', 'cxf-fediz', 'deltaspike', 'systemml', 'calcite',
#      'fineract', 'parquet-mr', 'knox', 'streams', 'tez', 'lens', 'zeppelin', 'samza', 'phoenix',
#      'directory-kerby', 'kylin', 'ranger']

# p = ['pig', 'manifoldcf',
#      'giraph', 'bigtop', 'kafka', 'oozie',
#      'falcon', 'deltaspike', 'calcite',
#      'parquet-mr', 'tez', 'lens', 'phoenix',
#      # 'kylin',
#      'ranger']

# p = ['mahout', 'pdfbox', 'opennlp', 'openwebbeans', 'mina-sshd', 'freemarker',
#      'gora', 'helix', 'curator', 'flume',
#      'directory-fortress-core', 'storm', 'cxf-fediz', 'systemml',
#      'fineract', 'knox', 'streams', 'zeppelin', 'samza',
#      'directory-kerby', 'commons-rdf', 'nifi', 'eagle']

# p = ['pig', 'manifoldcf',
#      'giraph', 'bigtop', 'kafka', 'oozie',
#      'falcon', 'deltaspike', 'calcite',
#      'parquet-mr', 'tez', 'lens', 'phoenix']

# p = ['kafka', 'pig', 'bigtop', 'phoenix']

# p = ['giraph']

p = ['mahout', 'pdfbox', 'opennlp', 'openwebbeans', 'mina-sshd', 'helix', 'curator',
     'storm', 'cxf-fediz',
     'knox', 'zeppelin', 'samza',
     'directory-kerby', 'pig', 'manifoldcf',
     'giraph', 'bigtop', 'kafka', 'oozie',
     'falcon', 'deltaspike', 'calcite',
     'parquet-mr', 'tez', 'lens', 'phoenix',
     'kylin', 'ranger']

csv_path = open('caliberated_performace_results.csv', 'a')

def evaluate(y_pred, y_test):
      # accuarcy, precision, recall, f1, roc_auc
      _accuracy_score = accuracy_score(y_test, y_pred)
      _precision_score = precision_score(y_test, y_pred)
      _recall_score = recall_score(y_test, y_pred)
      _f1_score = f1_score(y_test, y_pred)
      _roc_auc_score = roc_auc_score(y_test, y_pred)

      metrics_list = [_accuracy_score, _precision_score, _recall_score, _f1_score, _roc_auc_score]
      return metrics_list
    


for i in range(len(p)):
      print(f"{i}. {p[i]}")
      df_train = pd.read_csv(
            f"new_tim_features/{p[i]}_fatty_commits_test_without_sp_new_features_issue_old_src_new_time.csv")
      df_test = pd.read_csv(
            f"new_tim_features/{p[i]}_fatty_commits_test_without_sp_new_features_issue_old_src_new_time.csv")

      # columns = ['CN', 'TN', 'JC', 'AA', 'PA', 'SPL', 'F-I-F', 'F-A-F', 'F-C-F', 'F-M1-F', 'F-M2-F', 'F-M3-F', 'F-R-F',
      #            'F-I-D-I-F', 'F-D-I-F', 'F-A-I-F', 'F-I-C-I-F', 'F-I-A-I-F', 'F-I-R-I-F']

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
      # scaling = StandardScaler()
      # X_train = scaling.transform(X_train)
      # X_test = scaling.transform(X_test)

      print(X_train.shape)
      print(y_train.shape)
      print(X_test.shape)
      print(y_test.shape)

      print("Logistic Regression:")
      reg_log = LogisticRegression(max_iter=10000)
      cccv1 = CalibratedClassifierCV(reg_log, cv=5, method='isotonic')
      cccv1.fit(X_train, y_train)
      y_pre1 = cccv1.predict(X_test)
      #reg_log.fit(X_train, y_train)
      #y_pre1 = reg_log.predict(X_test)
      print(metrics.classification_report(y_test, y_pre1))
      print("roc_auc_score: ", roc_auc_score(y_test, y_pre1))
      print("f1 score: ", f1_score(y_test, y_pre1,
            average='weighted', labels=np.unique(y_pre1)))
      evaluate(y_pre1, y_test)

      print("\nRandom Forest:")
      reg_rf = RandomForestClassifier()
      cccv2 = CalibratedClassifierCV(reg_rf, cv=5, method='isotonic')
      cccv2.fit(X_train, y_train)
      y_pre2 = cccv2.predict(X_test)
      # reg_rf.fit(X_train, y_train)
      # y_pre2 = reg_rf.predict(X_test)
      print(metrics.classification_report(y_test, y_pre2))
      print("roc_auc_score: ", roc_auc_score(y_test, y_pre2))
      print("f1 score: ", f1_score(y_test, y_pre2,
            average='weighted', labels=np.unique(y_pre1)))
      evaluate(y_pre2, y_test)

      print("\nDecision Tree:")
      reg_dec = DecisionTreeClassifier()
      cccv3 = CalibratedClassifierCV(reg_dec, cv=5, method='isotonic')
      cccv3.fit(X_train, y_train)
      y_pre3 = cccv3.predict(X_test)
      # reg_dec.fit(X_train, y_train)
      # y_pre4 = reg_dec.predict(X_test)
      print(metrics.classification_report(y_test, y_pre3))
      print("roc_auc_score: ", roc_auc_score(y_test, y_pre3))
      print("f1 score: ", f1_score(y_test, y_pre3,
            average='weighted', labels=np.unique(y_pre1)))
      evaluate(y_pre3, y_test)

      print("\nNaive Bayes :")
      reg_bay = GaussianNB()
      cccv4 = CalibratedClassifierCV(reg_bay, cv=5, method='isotonic')
      cccv4.fit(X_train, y_train)
      y_pre4 = cccv4.predict(X_test)
      # reg_bay.fit(X_train, y_train)
      # y_pre5 = reg_bay.predict(X_test)
      print(metrics.classification_report(y_test, y_pre4))
      print("roc_auc_score: ", roc_auc_score(y_test, y_pre4))
      print("f1 score: ", f1_score(y_test, y_pre4,
            average='weighted', labels=np.unique(y_pre1)))
      evaluate(y_pre4, y_test)

      print("\nAdaboost Classifier:")
      reg_ada = AdaBoostClassifier()
      cccv5 = CalibratedClassifierCV(reg_ada, cv=5, method='isotonic')
      cccv5.fit(X_train, y_train)
      y_pre5 = cccv5.predict(X_test)
      # reg_ada.fit(X_train, y_train)
      # y_pre6 = reg_ada.predict(X_test)
      print(metrics.classification_report(y_test, y_pre5))
      print("roc_auc_score: ", roc_auc_score(y_test, y_pre5))
      print("f1 score: ", f1_score(y_test, y_pre5, average='weighted', labels=np.unique(y_pre1)))
      evaluate(y_pre5, y_test)
      
      print("\nGradient Boosting Classifier:")
      reg_gra = GradientBoostingClassifier()
      cccv6 = CalibratedClassifierCV(reg_gra, cv=5, method='isotonic')
      cccv6.fit(X_train, y_train)
      y_pre6 = cccv6.predict(X_test)
      # reg_gra.fit(X_train, y_train)
      # y_pre7 = reg_gra.predict(X_test)
      print(metrics.classification_report(y_test, y_pre6))
      print("roc_auc_score: ", roc_auc_score(y_test, y_pre6))
      print("f1 score: ", f1_score(y_test, y_pre6, average='weighted', labels=np.unique(y_pre1)))
      evaluate(y_pre6, y_test)

      # print("------------------------------------------------------------------------------------")

      # print("\nSupport Vector Machines (SVM):")
      # reg_svc = SVC(kernel='linear')
      # reg_svc.fit(X_train, y_train)
      # y_pre3 = reg_svc.predict(X_test)
      # print(metrics.classification_report(y_test, y_pre3))
      # print("roc_auc_score: ", roc_auc_score(y_test, y_pre3))
      # print("f1 score: ", f1_score(y_test, y_pre3))
      print("------------------------------------------------------------------------------------")
