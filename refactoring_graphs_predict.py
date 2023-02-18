# this is same as the refactoring_new but for predicted file pairs

# this code is to calculate the following stats:
# 1. refactoring predicted file pairs and commits corresponding to them
# 2. those commits distribution according to types
# 3. above stats for predicted intermodule file pairs
import math
import operator

import numpy
import pandas
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from pymongo import MongoClient
import networkx as nx
from datetime import datetime, timedelta
from itertools import combinations
from Build_reverse_identity_dictionary import Build_reverse_identity_dictionary
from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from bson import ObjectId
from sklearn.linear_model import LogisticRegression
from refactoring_new import Final


class final:

    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        # Access database
        self.db = self.client["smartshark"]
        # Access collection of the database
        self.BRID = Build_reverse_identity_dictionary()
        self.BRID.reading_identity_and_people_and_building_reverse_identity_dictionary()
        self.test_listp0 = list()
        self.test_listp1 = list()
        self.test_listps0 = list()
        self.test_listps1 = list()
        self.s41 = set()
        self.s40 = set()
        self.s40_fp = set()
        self.s41_fp = set()
        self.file_path = self.db["file"]
        self.file_path_records = list(self.file_path.find({}, {"_id": 1, "path": 1}))
        self.file_path_data = dict()
        self.file_level1 = dict()
        self.commit_file = self.db["file_action"]
        self.commit_file_records = list(self.commit_file.find({}, {"file_id": 1, "commit_id": 1}))
        self.commit_set_of_files = dict()
        self.commit_set_of_src_files = dict()
        self.commit_project = self.db["commit_with_project_info"]
        self.commit_project_records = list(self.commit_project.find({}, {}))
        self.commit_project_data = dict()
        self.refactoring_data = self.db["refactoring"]
        self.refactoring_records = list(self.refactoring_data.find({}, {}))
        self.commit_refactoring = dict()
        self.commit_refactoring_test = dict()
        self.refactor_type = dict()
        self.refactor_description = dict()
        self.commit_time_unsort = dict()
        self.commit_time = dict()
        self.test_data = dict()
        self.all_types = set()
        self.commit_type_test = dict()
        self.commit_description_test = dict()
        self.commits_in_s41_fp = set()
        self.type_frequency_unsort = dict()
        self.s41_intermodule_fp = set()
        self.s41_intermodule_refactor_fp = set()
        self.commits_in_s41_intermodule_refactor_fp = set()
        self.type_frequency_intermodule_unsort = dict()
        self.gpred = nx.DiGraph()
        self.testlistps = list()
        self.training_data = dict()
        self.gtrain = nx.Graph()
        self.gtest = nx.Graph()
        self.r = nx.Graph()
        self.ps = set()  # file pairs not connected in training
        self.ps1 = set()
        self.psf = set()  # unique files in ps
        self.commits_in_s41_tc_fp = set()
        self.s2 = set()
        self.s2_fp = set()
        self.commits_in_s2_fp = set()
        self.s = set()  # file pairs not connected in training but predicted to be connected in test
        self.g = nx.Graph()
        self.guw = nx.Graph()
        self.training_fp_frequency = dict()
        self.m1 = dict()
        self.m2 = dict()
        self.m3 = dict()
        self.snewm1 = set()
        self.snewm2 = set()
        self.snewm3 = set()
        self.s_commits = dict()
        self.s_fp_recall = dict()
        self.sm1_commits = dict()
        self.sm1_fp_recall = dict()
        self.sm2_commits = dict()
        self.sm2_fp_recall = dict()
        self.sm3_commits = dict()
        self.sm3_fp_recall = dict()
        self.s_files = set()
        self.s_file_commits = dict()
        self.s_precision = dict()
        self.sm1_files = set()
        self.sm1_file_commits = dict()
        self.sm1_precision = dict()
        self.sm2_files = set()
        self.sm2_file_commits = dict()
        self.sm2_precision = dict()
        self.sm3_files = set()
        self.sm3_file_commits = dict()
        self.sm3_precision = dict()
        self.refactoring_files_test = set()
        self.s_refactor_files = set()
        self.sm1_refactor_files = set()
        self.sm2_refactor_files = set()
        self.sm4_refactor_files = set()

    def build_commit_project_data_time(self):
        for element in self.commit_project_records:
            commit_id = element["_id"]
            project_name = element["project_name_info"]["name"]
            c_time = element["committer_date"]
            if commit_id not in self.commit_project_data.keys():
                self.commit_project_data[commit_id] = project_name
            if commit_id not in self.commit_time_unsort.keys():
                self.commit_time_unsort[commit_id] = c_time
        self.commit_time = {key: value for key, value in sorted(self.commit_time_unsort.items(),
                                                                key=lambda item: item[1])}

    def build_file_path(self):
        for element in self.file_path_records:
            file_id = element["_id"]
            path = element["path"]
            if file_id not in self.file_path_data.keys():
                self.file_path_data[file_id] = path

    def build_file_level1(self):
        for file in self.file_path_data.keys():
            path = self.file_path_data[file]
            pos = path.rfind('/')
            if pos != -1:
                trim = path[0:pos:1]
                modules = trim.split("/")
                sz = len(modules)
                if sz >= 1:
                    # print(f"path = {path} |level1 = {modules[sz - 1]}")
                    if file not in self.file_level1.keys():
                        self.file_level1[file] = list()
                    if modules[sz - 1] not in self.file_level1[file]:
                        self.file_level1[file].append(modules[sz - 1])

    def build_commit_set_of_files(self):
        for element in self.commit_file_records:
            file_id = element["file_id"]
            commit_id = element["commit_id"]
            if commit_id not in self.commit_set_of_files.keys():
                self.commit_set_of_files[commit_id] = set()
            self.commit_set_of_files[commit_id].add(file_id)

    def build_commit_set_of_src_files(self, project):
        for commit in self.commit_set_of_files:
            if self.commit_project_data[commit] == project:
                files = self.commit_set_of_files[commit]
                files = list(files)
                r = len(files)
                for i in range(r):
                    path1 = self.file_path_data[files[i]]
                    pos1 = path1.rfind('/')
                    if pos1 != -1:
                        trim1 = path1[0:pos1:1]
                        modules1 = trim1.split("/")
                        if "src" in modules1:
                            if commit not in self.commit_set_of_src_files.keys():
                                self.commit_set_of_src_files[commit] = list()
                            if i not in self.commit_set_of_src_files[commit]:
                                self.commit_set_of_src_files[commit].append(files[i])

    def build_commit_refactoring(self):
        for element in self.refactoring_records:
            commit_id = element["commit_id"]
            refactor_id = element["_id"]
            if commit_id not in self.commit_refactoring.keys():
                self.commit_refactoring[commit_id] = list()
            self.commit_refactoring[commit_id].append(refactor_id)

    def build_time_division(self, project):
        time_list = list()
        for commit in self.commit_time:
            if self.commit_project_data[commit] == project:
                # print(f"commit = {commit} | time = {self.commit_time[commit]}")
                time_list.append(self.commit_time[commit])
        t1min1 = min(time_list)
        t7max = max(time_list)
        print(f"Min time t1min1 = {t1min1}")
        print(f"Max time t7max = {t7max}")
        t13 = t1min1 + relativedelta(years=2)
        t1 = t13 + timedelta(days=1)
        t2m = t1 + relativedelta(months=12)
        t2d = t2m + relativedelta(days=18)
        t2 = t2d + timedelta(hours=6)
        t3 = t2 + timedelta(days=1)
        t4m = t3 + relativedelta(months=5)
        t4d = t4m + relativedelta(days=12)
        t4 = t4d + timedelta(hours=4)
        t5m = t3 + relativedelta(months=12)
        t5d = t5m + relativedelta(days=18)
        t5 = t5d + timedelta(hours=6)
        t6 = t5 + timedelta(days=1)
        t7m = t6 + relativedelta(months=5)
        t7d = t7m + relativedelta(days=12)
        t7 = t7d + timedelta(hours=4)
        print(f"t1 = {t1}")
        print(f"t2 = {t2}")
        print(f"t3 = {t3}")
        print(f"t4 = {t4}")
        print(f"t5 = {t5}")
        print(f"t6 = {t6}")
        print(f"t7 = {t7}")
        c = 0
        for commit in self.commit_time:
            if self.commit_project_data[commit] == project:
                if t1 <= self.commit_time[commit] <= t7:
                    c = c + 1
        print(f"Number of commits between t1 and t7 = {c}")
        c2 = 0
        for commit in self.commit_time:
            if self.commit_project_data[commit] == project:
                if t1 <= self.commit_time[commit] <= t2:
                    c2 = c2 + 1
        print(f"Number of commits between t1 and t2 : {c2}")
        c1 = 0
        for commit in self.commit_time:
            if self.commit_project_data[commit] == project:
                if t2 < self.commit_time[commit] <= t3:
                    c1 = c1 + 1
        print(f"Number of commits between t2 and t3 : {c1}")
        c3 = 0
        for commit in self.commit_time:
            if self.commit_project_data[commit] == project:
                if t3 < self.commit_time[commit] <= t4:
                    c3 = c3 + 1
        print(f"Number of commits between t3 and t4 : {c3}")
        c4 = 0
        for commit in self.commit_time:
            if self.commit_project_data[commit] == project:
                if t4 < self.commit_time[commit] <= t5:
                    c4 = c4 + 1
        print(f"Number of commits between t4 and t5 : {c4}")
        c5 = 0
        for commit in self.commit_time:
            if self.commit_project_data[commit] == project:
                if t5 < self.commit_time[commit] <= t6:
                    c5 = c5 + 1
        print(f"Number of commits between t5 and t6 : {c5}")
        c6 = 0
        for commit in self.commit_time:
            if self.commit_project_data[commit] == project:
                if t6 < self.commit_time[commit] <= t7:
                    c6 = c6 + 1
        print(f"Number of commits between t6 and t7 : {c6}")
        self.build_training_data(t3, t5, project)
        self.build_test_data(t6, t7, project)

    def build_training_data(self, t3, t5, project):
        self.training_data.clear()
        for commit in self.commit_time:
            if self.commit_project_data[commit] == project:
                if t3 <= self.commit_time[commit] <= t5:
                    self.training_data[commit] = self.commit_time[commit]

    def build_test_data(self, t6, t7, project):
        self.test_data.clear()
        for commit in self.commit_time:
            if self.commit_project_data[commit] == project:
                if t6 <= self.commit_time[commit] <= t7:
                    self.test_data[commit] = self.commit_time[commit]

    def build_training_file(self, project):
        self.gtrain.clear()
        for commit in self.commit_set_of_src_files:
            if self.commit_project_data[commit] == project:
                if commit in self.training_data:
                    files = self.commit_set_of_src_files[commit]
                    # if 30 >= len(files) >= 1:
                    ordered_pairs = {(x, y) for x in files for y in files if x != y}
                    self.gtrain.add_edges_from(ordered_pairs)
        print(f"Length of edges training data = {len(self.gtrain.edges())}")
        print(f"Length of nodes training data = {len(self.gtrain.nodes())}")

    def build_test_file(self, project):
        self.gtest.clear()
        for commit in self.commit_set_of_src_files:
            if self.commit_project_data[commit] == project:
                if commit in self.test_data:
                    files = self.commit_set_of_src_files[commit]
                    # if 30 >= len(files) >= 1:
                    ordered_pairs = {(x, y) for x in files for y in files if x != y}
                    self.gtest.add_edges_from(ordered_pairs)
        print(f"Length of edges test data = {len(self.gtest.edges())}")
        print(f"Length of nodes test data = {len(self.gtest.nodes())}")

    def build_r_graph(self):
        self.r = nx.intersection(self.gtrain, self.gtest)
        print(f"training nodes = {len(self.gtrain.nodes())}")
        print(f"test nodes = {len(self.gtest.nodes())}")
        print(f"common nodes in training and test = {len(self.r.nodes())}")
        unique_pairs = list()
        for i in combinations(self.r.nodes(), 2):
            unique_pairs.append(i)
        print(f"unique_pairs length = {len(unique_pairs)}")
        for d in unique_pairs:
            a, b = d
            if self.gtrain.has_edge(a, b):
                self.ps1.add(d)
        for d in unique_pairs:
            a, b = d
            if not self.gtrain.has_edge(a, b):
                self.ps.add(d)
        for d in self.ps:
            a, b = d
            self.psf.add(a)
            self.psf.add(b)
        for d in self.ps:
            a, b = d
            if self.gtest.has_edge(a, b):
                self.s2.add(d)
        for commit in self.commit_refactoring_test:
            if commit in self.commit_set_of_src_files:
                files = self.commit_set_of_src_files[commit]
                # if len(files) <= 30:
                for d in self.s2:
                    a, b = d
                    if a in files and b in files:
                        self.s2_fp.add(d)
                        self.commits_in_s2_fp.add(commit)

    def build_refactor_commits_test(self):
        for commit in self.commit_refactoring:
            if commit in self.test_data.keys():
                self.commit_refactoring_test[commit] = self.commit_refactoring[commit]
        print(f"length of refactoring commits = {len(self.commit_refactoring)}")
        print(f"length of refactoring commits in test period = {len(self.commit_refactoring_test)}")

    def build_type_description(self):
        for element in self.refactoring_records:
            refactor_id = element["_id"]
            r_type = element["type"]
            self.all_types.add(r_type)
            if refactor_id not in self.refactor_type.keys():
                self.refactor_type[refactor_id] = list()
            self.refactor_type[refactor_id].append(r_type)

        for element in self.refactoring_records:
            refactor_id = element["_id"]
            desc = element["description"]
            if refactor_id not in self.refactor_description.keys():
                self.refactor_description[refactor_id] = list()
            self.refactor_description[refactor_id].append(desc)

        for commit in self.commit_refactoring_test:
            refactor = self.commit_refactoring_test[commit]
            r = len(refactor)
            for i in range(r):
                ty = self.refactor_type[refactor[i]]
                des = self.refactor_description[refactor[i]]
                if commit not in self.commit_type_test.keys():
                    self.commit_type_test[commit] = list()
                self.commit_type_test[commit].append(ty)
                if commit not in self.commit_description_test.keys():
                    self.commit_description_test[commit] = list()
                self.commit_description_test[commit].append(des)

    def build_prediction(self, project):
        df_train = pd.read_csv(f"{project}_all_commits_src_training.csv")
        df_test = pd.read_csv(f"{project}_all_commits_src_test.csv")
        column1 = ['ID-a', 'ID-b']
        columns = ['CN', 'TN', 'JC', 'AA', 'PA', 'SPL', 'F-I-F', 'F-A-F', 'F-C-F', 'F-M1-F', 'F-M2-F', 'F-M3-F',
                   'F-R-F',
                   'F-I-D-I-F', 'F-D-I-F', 'F-A-I-F', 'F-I-C-I-F', 'F-I-A-I-F', 'F-I-R-I-F']
        # columns = ['CN', 'TN', 'JC', 'AA', 'PA', 'SPL']
        X_train = np.array(df_train[columns])
        y_train = np.array(df_train['CV'])
        X_test1 = np.array(df_test[columns])
        X_testl = np.array(df_test[column1])
        X_trainl = np.array(df_train[column1])
        y_test1 = np.array(df_test['CV'])

        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)

        print("\nLogistic Regression:")
        reg_rf = LogisticRegression(max_iter=10000)
        # reg_rf = RandomForestClassifier()
        reg_rf.fit(X_train, y_train)
        y_pre2 = reg_rf.predict(X_test1)
        print(metrics.classification_report(y_test1, y_pre2))
        CM = confusion_matrix(y_test1, y_pre2)
        print("roc_auc_score: ", roc_auc_score(y_test1, y_pre2))
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]

        print(f"true positives = {TP}")
        print(f"true negatives = {TN}")
        print(f"false positives = {FP}")
        print(f"false negatives = {FN}\n")

        for i in range(len(y_test1)):
            y = list()
            y0 = list()
            if y_pre2[i] == 1:
                y.clear()
                X1 = X_testl[i][0].split("\'")
                Xo1 = ObjectId(X1[1])
                y.append(Xo1)
                X2 = X_testl[i][1].split("\'")
                Xo2 = ObjectId(X2[1])
                y.append(Xo2)
                y = tuple(y)
                self.test_listp1.append(y)
            if y_pre2[i] == 0:
                y0.clear()
                X10 = X_testl[i][0].split("\'")
                Xo10 = ObjectId(X10[1])
                y0.append(Xo10)
                X20 = X_testl[i][1].split("\'")
                Xo20 = ObjectId(X20[1])
                y0.append(Xo20)
                y0 = tuple(y0)
                self.test_listp0.append(y0)

        for i in range(len(self.test_listp1)):
            path1 = self.file_path_data[self.test_listp1[i][0]]
            path2 = self.file_path_data[self.test_listp1[i][1]]
            pos1 = path1.rfind('/')
            pos2 = path2.rfind('/')
            if pos1 != -1 and pos2 != -1:
                trim1 = path1[0:pos1:1]
                modules1 = trim1.split("/")
                trim2 = path2[0:pos2:1]
                modules2 = trim2.split("/")
                if "src" in modules1 and "src" in modules2:
                    self.test_listps1.append(self.test_listp1[i])
                    self.testlistps.append(self.test_listp1[i])

        for i in range(len(self.test_listp0)):
            path1 = self.file_path_data[self.test_listp0[i][0]]
            path2 = self.file_path_data[self.test_listp0[i][1]]
            pos1 = path1.rfind('/')
            pos2 = path2.rfind('/')
            if pos1 != -1 and pos2 != -1:
                trim1 = path1[0:pos1:1]
                modules1 = trim1.split("/")
                trim2 = path2[0:pos2:1]
                modules2 = trim2.split("/")
                if "src" in modules1 and "src" in modules2:
                    self.test_listps0.append(self.test_listp0[i])
                    self.testlistps.append(self.test_listp0[i])

        self.gpred.add_edges_from(self.test_listps1)

        for i in range(len(self.test_listps1)):
            d = self.test_listps1[i]
            a, b = d
            self.s41.add(d)

        print(f"len of predicted pairs 1 = {len(self.s41)}")

        for i in range(len(self.test_listps0)):
            d = self.test_listps0[i]
            a, b = d
            self.s40.add(d)

        print(f"len of predicted pairs 0 = {len(self.s40)}\n")

        # self.s41_fp = set()
        # for commit in self.commit_refactoring_test:
        #     if commit in self.commit_set_of_src_files:
        #         files = self.commit_set_of_src_files[commit]
        #         # if len(files) <= 30:
        #         for d in self.s41:
        #             a, b = d
        #             if a in files and b in files:
        #                 self.s41_fp.add(d)
        # print(f"file pairs in s41 = {len(self.s41)}")
        # print(f"refactoring file pairs in s41 = {len(self.s41_fp)}")
        #
        # self.s40_fp = set()
        # for commit in self.commit_refactoring_test:
        #     if commit in self.commit_set_of_src_files:
        #         files = self.commit_set_of_src_files[commit]
        #         # if len(files) <= 30:
        #         for d in self.s40:
        #             a, b = d
        #             if a in files and b in files:
        #                 self.s40_fp.add(d)
        # print(f"file pairs in s40 = {len(self.s40)}")
        # print(f"refactoring file pairs in s40 = {len(self.s40_fp)}\n")
        #
        # commit_file_pair_s41 = dict()
        # c3 = 0
        # for commit in self.commit_refactoring_test:
        #     if commit in self.commit_set_of_src_files:
        #         files = self.commit_set_of_src_files[commit]
        #         # if len(files) <= 30:
        #         for d in self.s41:
        #             a, b = d
        #             if a in files and b in files:
        #                 if d not in commit_file_pair_s41.keys():
        #                     commit_file_pair_s41[d] = list()
        #                 if commit not in commit_file_pair_s41[d]:
        #                     commit_file_pair_s41[d].append(commit)
        #                 self.commits_in_s41_fp.add(commit)

        for d in self.s41:
            a, b = d
            if not self.gtrain.has_edge(a, b):
                self.s.add(d)

        print(f"file pairs not connected in training but predicted to be connected in test = {len(self.s)}")

        for d in self.ps1:
            a, b = d
            c = 0
            for commit in self.commit_set_of_src_files.keys():
                if commit in self.training_data:
                    files = self.commit_set_of_src_files[commit]
                    if a in files and b in files:
                        c = c + 1
            if c != 0:
                self.training_fp_frequency[d] = c

        for f in self.training_fp_frequency:
            freq = self.training_fp_frequency[f]
            a, b = f
            # print(f"a = {a}")
            # print(f"b = {b}")
            # print(f"freq = {freq}")
            tu = (a, b, freq)
            a = str(a)
            b = str(b)
            self.g.add_edge(a, b, weight=freq)
            self.guw.add_edge(a, b)

        print(f"len of g nodes = {len(self.g.nodes())}")
        print(f"len of r nodes = {len(self.r.nodes())}")  # both are not equal as some frequncies are 0

        paths = nx.johnson(self.g, weight="weight")

        li1 = list()
        for i in self.g.nodes():
            li1.append(i)
        #
        index = 0
        for d in self.s:
            index = index + 1
            print(f"{index}")
            dest = dict()
            self.m1.clear()
            self.m2.clear()
            self.m3.clear()
            a, b = d
            # print(f"\nfp = {d} | type = {type(d)}")
            a = str(a)
            b = str(b)
            # print(f"a = {a} | type = {type(a)}")
            for j in li1:
                # print(f"j = {j} | type = {type(j)}")
                dest[j] = list()
                if a in li1:
                    # print(paths[a][j])
                    patha = paths[a][j]
                    pla = len(patha) - 1
                    # patha = nx.dijkstra_path(self.guw, a, j)
                    # pla = nx.dijkstra_path_length(self.guw, a, j)
                    x1 = (a, patha, pla)
                    # print(f"x1 = {x1}")
                    dest[j].append(x1)
                if b in li1:
                    pathb = paths[b][j]
                    plb = len(pathb) - 1
                    # pathb = nx.dijkstra_path(self.guw, b, j)
                    # plb = nx.dijkstra_path_length(self.guw, b, j)
                    x2 = (b, pathb, plb)
                    dest[j].append(x2)
            # for k in dest:
            #     print(f"N = {k} | list = {dest[k]}")
            dest1 = dict()
            for N in dest.keys():
                li2 = dest[N]
                li2a = li2[0]
                li2b = li2[1]
                Na = li2a[0]
                Npatha = li2a[1]
                Npathlengtha = li2a[2]
                Nb = li2b[0]
                Npathb = li2b[1]
                Npathlengthb = li2b[2]
                Nwa = nx.path_weight(self.g, Npatha, 'weight')
                Nwb = nx.path_weight(self.g, Npathb, 'weight')
                Nx1 = (Na, Npathlengtha, Nwa)
                Nx2 = (Nb, Npathlengthb, Nwb)
                dest1[N] = list()
                dest1[N].append(Nx1)
                dest1[N].append(Nx2)
            # for N in dest1.keys():
            #     print(f"N = {N} | list = {dest1[N]}")
            for N1 in dest1.keys():
                li3 = dest1[N1]
                li3a = li3[0]
                li3b = li3[1]
                N1a = li3a[0]
                N1pathlengtha = li3a[1]
                N1weighta = li3a[2]
                N1b = li3b[0]
                N1pathlengthb = li3b[1]
                N1weightb = li3b[2]
                if N1pathlengtha == 1 and N1pathlengthb == 1:
                    M1 = N1weighta + N1weightb
                else:
                    M1 = -100
                if N1pathlengtha == 2 and N1pathlengthb == 2:
                    M2 = N1weighta + N1weightb
                else:
                    M2 = -100
                if N1pathlengtha == 3 and N1pathlengthb == 3:
                    M3 = N1weighta + N1weightb
                else:
                    M3 = -100
                if M1 != -100:
                    self.m1[N1] = M1
                if M2 != -100:
                    self.m2[N1] = M2
                if M3 != -100:
                    self.m3[N1] = M3
            self.m1 = dict(sorted(self.m1.items(), key=operator.itemgetter(1), reverse=True)[:5])
            #  sorted(A, key=A.get, reverse=True)[:5]
            self.m2 = dict(sorted(self.m2.items(), key=operator.itemgetter(1), reverse=True)[:5])
            self.m3 = dict(sorted(self.m3.items(), key=operator.itemgetter(1), reverse=True)[:5])
            m1nodes = set()
            m2nodes = set()
            m3nodes = set()
            a = ObjectId(a)
            b = ObjectId(b)
            for N1 in self.m1.keys():
                N1 = ObjectId(N1)
                m1nodes.add(a)
                m1nodes.add(b)
                m1nodes.add(N1)
                m1tuple = tuple(m1nodes)
                # print(f"node = {N1} | M1 = {self.m1[N1]}")
            for N1 in self.m2.keys():
                N1 = ObjectId(N1)
                m2nodes.add(a)
                m2nodes.add(b)
                m2nodes.add(N1)
                m2tuple = tuple(m2nodes)
                # print(f"node = {N1} | M2 = {self.m2[N1]}")
            for N1 in self.m3.keys():
                N1 = ObjectId(N1)
                m3nodes.add(a)
                m3nodes.add(b)
                m3nodes.add(N1)
                m3tuple = tuple(m3nodes)
                # print(f"node = {N1} | M3 = {self.m3[N1]}")
            if m1tuple:
                # print(f"m1 tuples = {m1tuple}")
                self.snewm1.add(m1tuple)
            # print(f"m2 tuples = {m2tuple}")
            # if m2tuple:
            self.snewm2.add(m2tuple)
            # print(f"m3 tuples = {m3tuple}")
            # if m3tuple:
            self.snewm3.add(m3tuple)

        for d in self.s:
            a, b = d
            for commit in self.commit_refactoring_test.keys():
                if commit in self.commit_set_of_src_files:
                    files = self.commit_set_of_src_files[commit]
                    if a in files and b in files:
                        if d not in self.s_commits.keys():
                            self.s_commits[d] = list()
                        if commit not in self.s_commits[d]:
                            self.s_commits[d].append(commit)

        print(f"length of set s = {len(self.s)}")
        print(f"length of file pairs part of refactoring commits = {len(self.s_commits)}")
        # for d in self.s_commits:
        #     print(f"file pair = {d} | refactoring commits = {self.s_commits[d]}")

        # for d in self.s_commits:
        #     commits = self.s_commits[d]
        #     r = len(commits)
        #     for i in range(r):
        #         files = self.commit_set_of_src_files[commits[i]]
        #         print(f"commit = {commits[i]} | size = {len(files)}")

        for d in self.s_commits:
            rc = 0
            c = 0
            a, b = d
            commits = self.s_commits[d]
            r = len(commits)
            for i in range(r):
                files = self.commit_set_of_src_files[commits[i]]
                if a in files and b in files:
                    rc = rc + (len(d) / len(files))
                    c = c + 1
            recall = rc / c
            if d not in self.s_fp_recall.keys():
                self.s_fp_recall[d] = recall

        avgrecall = 0
        for d in self.s_fp_recall:
            avgrecall = avgrecall + self.s_fp_recall[d]
        print(f"average recall of s = {avgrecall / len(self.s_fp_recall)}")

        for d in self.snewm1:
            for commit in self.commit_refactoring_test.keys():
                if commit in self.commit_set_of_src_files:
                    files = self.commit_set_of_src_files[commit]
                    if set(d).issubset(files):
                        if d not in self.sm1_commits.keys():
                            self.sm1_commits[d] = list()
                        if commit not in self.sm1_commits[d]:
                            self.sm1_commits[d].append(commit)

        print(f"length of set snewm1 = {len(self.snewm1)}")
        print(f"length of file pairs part of refactoring commits = {len(self.sm1_commits)}")
        # for d in self.sm1_commits:
        #     print(f"file pair = {d} | refactoring commits = {self.sm1_commits[d]}")

        for d in self.sm1_commits:
            rc = 0
            c = 0
            commits = self.sm1_commits[d]
            r = len(commits)
            for i in range(r):
                files = self.commit_set_of_src_files[commits[i]]
                if set(d).issubset(files):
                    rc = rc + (len(d) / len(files))
                    c = c + 1
            recall = rc / c
            if d not in self.sm1_fp_recall.keys():
                self.sm1_fp_recall[d] = recall

        avgrecallm1 = 0
        for d in self.sm1_fp_recall:
            avgrecallm1 = avgrecallm1 + self.sm1_fp_recall[d]
        print(f"average recall of snewm1 = {avgrecallm1 / len(self.sm1_fp_recall)}")

        for d in self.snewm2:
            for commit in self.commit_refactoring_test.keys():
                if commit in self.commit_set_of_src_files:
                    files = self.commit_set_of_src_files[commit]
                    if set(d).issubset(files):
                        if d not in self.sm2_commits.keys():
                            self.sm2_commits[d] = list()
                        if commit not in self.sm2_commits[d]:
                            self.sm2_commits[d].append(commit)

        print(f"length of set snewm2 = {len(self.snewm2)}")
        print(f"length of file pairs part of refactoring commits = {len(self.sm2_commits)}")
        # for d in self.sm2_commits:
        #     print(f"file pair = {d} | refactoring commits = {self.sm2_commits[d]}")

        for d in self.sm2_commits:
            rc = 0
            c = 0
            commits = self.sm2_commits[d]
            r = len(commits)
            for i in range(r):
                files = self.commit_set_of_src_files[commits[i]]
                if set(d).issubset(files):
                    rc = rc + (len(d) / len(files))
                    c = c + 1
            recall = rc / c
            if d not in self.sm2_fp_recall.keys():
                self.sm2_fp_recall[d] = recall

        avgrecallm2 = 0
        for d in self.sm2_fp_recall:
            avgrecallm2 = avgrecallm2 + self.sm2_fp_recall[d]
        print(f"average recall of snewm2 = {avgrecallm2 / len(self.sm2_fp_recall)}")

        for d in self.snewm3:
            for commit in self.commit_refactoring_test.keys():
                if commit in self.commit_set_of_src_files:
                    files = self.commit_set_of_src_files[commit]
                    if set(d).issubset(files):
                        if d not in self.sm3_commits.keys():
                            self.sm3_commits[d] = list()
                        if commit not in self.sm3_commits[d]:
                            self.sm3_commits[d].append(commit)

        print(f"length of set snewm3 = {len(self.snewm3)}")
        print(f"length of file pairs part of refactoring commits = {len(self.sm3_commits)}")
        # for d in self.sm3_commits:
        #     print(f"file pair = {d} | refactoring commits = {self.sm3_commits[d]}")

        for d in self.sm3_commits:
            rc = 0
            c = 0
            commits = self.sm3_commits[d]
            r = len(commits)
            for i in range(r):
                files = self.commit_set_of_src_files[commits[i]]
                if set(d).issubset(files):
                    rc = rc + (len(d) / len(files))
                    c = c + 1
            recall = rc / c
            if d not in self.sm3_fp_recall.keys():
                self.sm3_fp_recall[d] = recall

        avgrecallm3 = 0
        for d in self.sm3_fp_recall:
            avgrecallm3 = avgrecallm3 + self.sm3_fp_recall[d]
        print(f"average recall of snewm3 = {avgrecallm3 / len(self.sm3_fp_recall)}")

        for f in self.s_commits:
            pr = 0
            c = 0
            d1 = set(f)
            commits = self.s_commits[f]
            r = len(commits)
            for i in range(r):
                files = self.commit_set_of_src_files[commits[i]]
                files = set(files)
                if files.intersection(d1):
                    pr = pr + (len(files.intersection(d1)) / len(files))
                    c = c + 1
            precision = pr / c
            if f not in self.s_precision.keys():
                self.s_precision[f] = precision

        avgpre = 0
        for f in self.s_precision:
            avgpre = avgpre + self.s_precision[f]
        print(f"average precision of s = {avgpre / len(self.s_precision)}")

        for f in self.sm1_commits:
            pr = 0
            c = 0
            d1 = set(f)
            commits = self.sm1_commits[f]
            r = len(commits)
            for i in range(r):
                files = self.commit_set_of_src_files[commits[i]]
                files = set(files)
                if files.intersection(d1):
                    pr = pr + (len(files.intersection(d1)) / len(files))
                    c = c + 1
            precision = pr / c
            if f not in self.sm1_precision.keys():
                self.sm1_precision[f] = precision

        avgprem1 = 0
        for f in self.sm1_precision:
            avgprem1 = avgprem1 + self.sm1_precision[f]
        print(f"average precision of sm1 = {avgprem1 / len(self.sm1_precision)}")

        for f in self.sm2_commits:
            pr = 0
            c = 0
            d1 = set(f)
            commits = self.sm2_commits[f]
            r = len(commits)
            for i in range(r):
                files = self.commit_set_of_src_files[commits[i]]
                files = set(files)
                if files.intersection(d1):
                    pr = pr + (len(files.intersection(d1)) / len(files))
                    c = c + 1
            precision = pr / c
            if f not in self.sm2_precision.keys():
                self.sm2_precision[f] = precision

        avgprem2 = 0
        for f in self.sm2_precision:
            avgprem2 = avgprem2 + self.sm2_precision[f]
        print(f"average precision of sm2 = {avgprem2 / len(self.sm2_precision)}")

        for f in self.sm3_file_commits:
            pr = 0
            c = 0
            d1 = set(f)
            commits = self.sm3_file_commits[f]
            r = len(commits)
            for i in range(r):
                files = self.commit_set_of_src_files[commits[i]]
                files = set(files)
                if files.intersection(d1):
                    pr = pr + (len(files.intersection(d1)) / len(files))
                    c = c + 1
            if c != 0:
                precision = pr / c
            else:
                precision = 0
            if f not in self.sm3_precision.keys():
                self.sm3_precision[f] = precision

        avgprem3 = 0
        for f in self.sm3_precision:
            avgprem3 = avgprem3 + self.sm3_precision[f]
        if len(self.sm3_precision) != 0:
            print(f"average precision of sm3 = {avgprem3 / len(self.sm3_precision)}")
        else:
            print(f"average precision of sm3 = {0}")

        for commit in self.commit_refactoring_test:
            if commit in self.commit_set_of_src_files.keys():
                files = self.commit_set_of_src_files[commit]
                r = len(files)
                for i in range(r):
                    self.refactoring_files_test.add(files[i])

        print(f"\nLength of refactoring files in test = {len(self.refactoring_files_test)}")

        for d in self.s:
            a, b = d
            self.s_files.add(a)
            self.s_files.add(b)

        for file in self.refactoring_files_test:
            if file in self.s_files:
                self.s_refactor_files.add(file)

        print(f"\nlength of files in s = {len(self.s_files)}")
        print(f"length of refactor files in s = {len(self.s_refactor_files)}")

        for d in self.snewm1:
            for i in d:
                self.sm1_files.add(i)

        for file in self.refactoring_files_test:
            if file in self.sm1_files:
                self.sm1_refactor_files.add(file)

        print(f"\nlength of files in sm1 = {len(self.sm1_files)}")
        print(f"length of refactor files in sm1 = {len(self.sm1_refactor_files)}")

        for d in self.snewm2:
            for i in d:
                self.sm2_files.add(i)

        for file in self.refactoring_files_test:
            if file in self.sm2_files:
                self.sm2_refactor_files.add(file)

        print(f"\nlength of files in sm2 = {len(self.sm2_files)}")
        print(f"length of refactor files in sm2 = {len(self.sm2_refactor_files)}")

        for d in self.snewm3:
            for i in d:
                self.sm3_files.add(i)

        for file in self.refactoring_files_test:
            if file in self.sm3_files:
                self.sm4_refactor_files.add(file)

        print(f"\nlength of files in sm3 = {len(self.sm3_files)}")
        print(f"length of refactor files in sm3 = {len(self.sm4_refactor_files)}")

        m1m2 = self.sm1_files.union(self.sm2_files)
        m1m2m3 = m1m2.union(self.sm3_files)
        print(f"\nLength of union of m1 m2 m3 sets = {len(m1m2m3)}")
        m1m2refactor = self.sm1_refactor_files.union(self.sm2_refactor_files)
        m1m2m3refactor = m1m2refactor.union(self.sm4_refactor_files)
        print(f"\nLength of union of refactor files of m1 m2 m3 = {len(m1m2m3refactor)}")


if __name__ == "__main__":
    st = datetime.now()
    print(f"start time : {datetime.now()}")
    t = final()
    print("\nCommit set of files")
    t.build_commit_set_of_files()
    print("\nCommit project and time building")
    t.build_commit_project_data_time()
    print("\nfile set of paths building")
    t.build_file_path()
    print("\nfile level 1 modules building")
    t.build_file_level1()
    p = "kafka"
    print("\nCommit set of src files building")
    t.build_commit_set_of_src_files(p)
    print("\ncommit refactoring building")
    t.build_commit_refactoring()
    print("\ntrain and test data building")
    t.build_time_division(p)
    print("\ntraining file building")
    t.build_training_file(p)
    print("\ntest file building")
    t.build_test_file(p)
    print("\nr graph building")
    t.build_r_graph()
    print("\nrefactoring commits in test building")
    t.build_refactor_commits_test()
    print("\nfile pair refactor type and description building")
    t.build_type_description()
    print("\npredicted files building")
    t.build_prediction(p)
    print(f"Start time : {st}")
    print(f"End time : {datetime.now()}")
