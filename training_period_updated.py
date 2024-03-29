import itertools
from dateutil.relativedelta import relativedelta
from pymongo import MongoClient
import networkx as nx
from datetime import datetime, timedelta
from itertools import combinations
from Build_reverse_identity_dictionary import Build_reverse_identity_dictionary
import csv
import os
import concurrent.futures
import matplotlib.pyplot as plt


class Task2:

    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["smartshark"]
        self.commit_project = self.db["commit_with_project_info"]
        self.commit_project_records = list(self.commit_project.find({}, {}))
        self.commit_file = self.db["file_action"]
        self.commit_refactoring = self.db["refactoring"]
        self.issue_author = self.db["issue_comment"]
        self.issue_reporter1 = self.db["issue"]
        self.file_path = self.db["file"]
        self.commit_issue_records = list(self.commit_project.find({}, {"_id": 1, "linked_issue_ids": 1}))
        self.commit_file_records = list(self.commit_file.find({}, {"file_id": 1, "commit_id": 1}))
        self.commit_refactor_records = list(self.commit_refactoring.find({}, {"_id": 1, "commit_id": 1}))
        self.issue_author_records = list(self.issue_author.find({}, {"issue_id": 1, "author_id": 1}))
        self.issue_reporter_records = list(self.issue_reporter1.find({}, {"_id": 1, "reporter_id": 1}))
        self.issue_assignee_records = list(self.issue_reporter1.find({}, {"_id": 1, "assignee_id": 1}))
        self.issue_component_records = list(self.issue_reporter1.find({}, {"_id": 1, "components": 1}))
        self.file_path_records = list(self.file_path.find({}, {"_id": 1, "path": 1}))
        self.BRID = Build_reverse_identity_dictionary()
        self.BRID.reading_identity_and_people_and_building_reverse_identity_dictionary()
        self.commit_set_of_files = dict()
        self.commit_project_data = dict()
        self.commit_time_unsort = dict()
        self.commit_time = dict()
        self.training_data1 = dict()
        self.training_data2 = dict()
        self.g1 = nx.Graph()
        self.g2 = nx.Graph()
        self.s = list()
        self.s1 = list()
        self.s2 = list()
        self.s3 = list()
        self.link_common_neighbors = dict()
        self.link_common_neighbors_union = dict()
        self.link_jc = dict()
        self.link_aa = dict()
        self.link_pa = dict()
        self.link_shortest_path = dict()
        self.link_class_variable = dict()
        self.commit_issue_data = dict()
        self.file_commits_data = dict()
        self.file_issue = dict()
        self.link_f_i_f = dict()
        self.commit_author_data = dict()
        self.file_developer_2a = dict()
        self.link_f_a_f = dict()
        self.commit_committer_data = dict()
        self.file_developer2b = dict()
        self.link_f_c_f = dict()
        self.file_path_data = dict()
        self.file_level1 = dict()
        self.link_f_l1_f = dict()
        self.file_level2 = dict()
        self.link_f_l2_f = dict()
        self.file_level3 = dict()
        self.link_f_l3_f = dict()
        self.commit_refactor_data = dict()
        self.file_developer4 = dict()
        self.link_f_r_f = dict()
        self.file_developer5 = dict()
        self.developer5_issue = dict()
        self.file_issue5 = dict()
        self.file_issue5_unique = dict()
        self.link_f_d_i_d_f = dict()
        self.issue_developer = dict()
        self.file_developer6 = dict()
        self.file_developer6_unique = dict()
        self.link_f_i_d_i_f = dict()
        self.issue_assignee = dict()
        self.file_assignee = dict()
        self.file_assignee_unique = dict()
        self.file_assignee_mapping = dict()
        self.file_assignee_mapping_unique = dict()
        self.link_f_d_i_f = dict()
        self.issue_developer8 = dict()
        self.file_developer8 = dict()
        self.file_developer8_unique = dict()
        self.file_developer8_mapping = dict()
        self.file_developer8_mapping_unique = dict()
        self.link_f_a_i_f = dict()
        self.issue_component = dict()
        self.file_component = dict()
        self.file_component_unique = dict()
        self.link_f_i_c_i_f = dict()
        self.file_developer11 = dict()
        self.file_developer11_unique = dict()
        self.link_f_i_a_i_f = dict()
        self.issue_reporter = dict()
        self.file_developer12 = dict()
        self.file_developer12_unique = dict()
        self.link_f_i_r_i_f = dict()
        self.commit_set_of_src_files = dict()

    def path_count12(self, f1, f2):
        developer_count_1 = dict()
        developer_count_2 = dict()
        path_count_var = 0

        if f1 in self.file_developer12.keys():
            for developer in self.file_developer12[f1]:
                if developer not in developer_count_1.keys():
                    developer_count_1[developer] = 1
                else:
                    developer_count_1[developer] = developer_count_1[developer] + 1
        if f2 in self.file_developer12.keys():
            for developer in self.file_developer12[f2]:
                if developer not in developer_count_2.keys():
                    developer_count_2[developer] = 1
                else:
                    developer_count_2[developer] = developer_count_2[developer] + 1

        for key in developer_count_1.keys():
            count1 = developer_count_1[key]
            if key in developer_count_2.keys():
                count2 = developer_count_2[key]
            else:
                count2 = 0
            path_count_var: int = path_count_var + count1 * count2

        return path_count_var

    def path_count11(self, f1, f2):
        developer_count_1 = dict()
        developer_count_2 = dict()
        path_count_var = 0

        if f1 in self.file_developer11.keys():
            for developer in self.file_developer11[f1]:
                if developer not in developer_count_1.keys():
                    developer_count_1[developer] = 1
                else:
                    developer_count_1[developer] = developer_count_1[developer] + 1
        if f2 in self.file_developer11.keys():
            for developer in self.file_developer11[f2]:
                if developer not in developer_count_2.keys():
                    developer_count_2[developer] = 1
                else:
                    developer_count_2[developer] = developer_count_2[developer] + 1

        for key in developer_count_1.keys():
            count1 = developer_count_1[key]
            if key in developer_count_2.keys():
                count2 = developer_count_2[key]
            else:
                count2 = 0
            path_count_var = path_count_var + (count1 * count2)

        return path_count_var

    def path_count9(self, f1, f2):
        developer_count_1 = dict()
        developer_count_2 = dict()
        path_count_var = 0

        if f1 in self.file_component.keys():
            for developer in self.file_component[f1]:
                if developer not in developer_count_1.keys():
                    developer_count_1[developer] = 1
                else:
                    developer_count_1[developer] = developer_count_1[developer] + 1
        if f2 in self.file_component.keys():
            for developer in self.file_component[f2]:
                if developer not in developer_count_2.keys():
                    developer_count_2[developer] = 1
                else:
                    developer_count_2[developer] = developer_count_2[developer] + 1

        for key in developer_count_1.keys():
            count1 = developer_count_1[key]
            if key in developer_count_2.keys():
                count2 = developer_count_2[key]
            else:
                count2 = 0
            path_count_var: int = path_count_var + count1 * count2

        return path_count_var

    def path_count8(self, f1, f2):
        developer_count = dict()
        developer_mapping_count = dict()
        path_count_var = 0

        if f1 in self.file_developer8.keys():
            for developer in self.file_developer8[f1]:
                if developer not in developer_count.keys():
                    developer_count[developer] = 1
                else:
                    developer_count[developer] = developer_count[developer] + 1

        if f2 in self.file_developer8_mapping.keys():
            for developer in self.file_developer8_mapping[f2]:
                if developer not in developer_mapping_count.keys():
                    developer_mapping_count[developer] = 1
                else:
                    developer_mapping_count[developer] = developer_mapping_count[developer] + 1

        for key in developer_count.keys():
            count1 = developer_count[key]
            if key in developer_mapping_count.keys():
                count2 = developer_mapping_count[key]
            else:
                count2 = 0
            path_count_var = path_count_var + count1 * count2

        return path_count_var

    def path_count7(self, f1, f2):
        assignee_count = dict()
        assignee_mapping_count = dict()
        path_count_var = 0

        if f1 in self.file_assignee.keys():
            for assignee in self.file_assignee[f1]:
                if assignee not in assignee_count.keys():
                    assignee_count[assignee] = 1
                else:
                    assignee_count[assignee] = assignee_count[assignee] + 1

        if f2 in self.file_assignee_mapping.keys():
            for assignee in self.file_assignee_mapping[f2]:
                if assignee not in assignee_mapping_count.keys():
                    assignee_mapping_count[assignee] = 1
                else:
                    assignee_mapping_count[assignee] = assignee_mapping_count[assignee] + 1

        for key in assignee_count.keys():
            count1 = assignee_count[key]
            if key in assignee_mapping_count.keys():
                count2 = assignee_mapping_count[key]
            else:
                count2 = 0
            path_count_var: int = path_count_var + count1 * count2

        return path_count_var

    def path_count6(self, f1, f2):
        developer_count_1 = dict()
        developer_count_2 = dict()
        path_count_var = 0

        if f1 in self.file_developer6.keys():
            for developer in self.file_developer6[f1]:
                if developer not in developer_count_1.keys():
                    developer_count_1[developer] = 1
                else:
                    developer_count_1[developer] = developer_count_1[developer] + 1
        if f2 in self.file_developer6.keys():
            for developer in self.file_developer6[f2]:
                if developer not in developer_count_2.keys():
                    developer_count_2[developer] = 1
                else:
                    developer_count_2[developer] = developer_count_2[developer] + 1

        for key in developer_count_1.keys():
            count1 = developer_count_1[key]
            if key in developer_count_2.keys():
                count2 = developer_count_2[key]
            else:
                count2 = 0
            path_count_var: int = path_count_var + count1 * count2

        return path_count_var

    def path_count5(self, f1, f2):
        issue_count_1 = dict()
        issue_count_2 = dict()
        path_count_var = 0

        if f1 in self.file_issue5.keys():
            for issue in self.file_issue5[f1]:
                if issue not in issue_count_1.keys():
                    issue_count_1[issue] = 1
                else:
                    issue_count_1[issue] = issue_count_1[issue] + 1
        if f2 in self.file_issue5.keys():
            for issue in self.file_issue5[f2]:
                if issue not in issue_count_2.keys():
                    issue_count_2[issue] = 1
                else:
                    issue_count_2[issue] = issue_count_2[issue] + 1

        for key in issue_count_1.keys():
            count1 = issue_count_1[key]
            if key in issue_count_2.keys():
                count2 = issue_count_2[key]
            else:
                count2 = 0
            path_count_var: int = path_count_var + count1 * count2

        return path_count_var

    def path_count4(self, f1, f2):
        developer_count_1 = dict()
        developer_count_2 = dict()
        path_count_var = 0

        if f1 in self.file_developer4.keys():
            for developer in self.file_developer4[f1]:
                if developer not in developer_count_1.keys():
                    developer_count_1[developer] = 1
                else:
                    developer_count_1[developer] = developer_count_1[developer] + 1
        if f2 in self.file_developer4.keys():
            for developer in self.file_developer4[f2]:
                if developer not in developer_count_2.keys():
                    developer_count_2[developer] = 1
                else:
                    developer_count_2[developer] = developer_count_2[developer] + 1

        for key in developer_count_1.keys():
            count1 = developer_count_1[key]
            if key in developer_count_2.keys():
                count2 = developer_count_2[key]
            else:
                count2 = 0
            path_count_var: int = path_count_var + count1 * count2

        return path_count_var

    def path_count3c(self, f1, f2):
        level3_count1 = dict()
        level3_count2 = dict()
        path_count_var = 0

        if f1 in self.file_level3.keys():
            for level3 in self.file_level3[f1]:
                if level3 not in level3_count1.keys():
                    level3_count1[level3] = 1
                else:
                    level3_count1[level3] = level3_count1[level3] + 1

        if f2 in self.file_level3.keys():
            for level3 in self.file_level3[f2]:
                if level3 not in level3_count2.keys():
                    level3_count2[level3] = 1
                else:
                    level3_count2[level3] = level3_count2[level3] + 1

        for key in level3_count1.keys():
            count1 = level3_count1[key]
            if key in level3_count2.keys():
                count2 = level3_count2[key]
            else:
                count2 = 0
            path_count_var: int = path_count_var + count1 * count2

        return path_count_var

    def path_count3b(self, f1, f2):
        level2_count1 = dict()
        level2_count2 = dict()
        path_count_var = 0

        if f1 in self.file_level2.keys():
            for level2 in self.file_level2[f1]:
                if level2 not in level2_count1.keys():
                    level2_count1[level2] = 1
                else:
                    level2_count1[level2] = level2_count1[level2] + 1

        if f2 in self.file_level2.keys():
            for level2 in self.file_level2[f2]:
                if level2 not in level2_count2.keys():
                    level2_count2[level2] = 1
                else:
                    level2_count2[level2] = level2_count2[level2] + 1

        for key in level2_count1.keys():
            count1 = level2_count1[key]
            if key in level2_count2.keys():
                count2 = level2_count2[key]
            else:
                count2 = 0
            path_count_var: int = path_count_var + count1 * count2

        return path_count_var

    def path_count3a(self, f1, f2):
        level1_count1 = dict()
        level1_count2 = dict()
        path_count_var = 0

        if f1 in self.file_level1.keys():
            for level1 in self.file_level1[f1]:
                if level1 not in level1_count1.keys():
                    level1_count1[level1] = 1
                else:
                    level1_count1[level1] = level1_count1[level1] + 1

        if f2 in self.file_level1.keys():
            for level1 in self.file_level1[f2]:
                if level1 not in level1_count2.keys():
                    level1_count2[level1] = 1
                else:
                    level1_count2[level1] = level1_count2[level1] + 1

        for key in level1_count1.keys():
            count1 = level1_count1[key]
            if key in level1_count2.keys():
                count2 = level1_count2[key]
            else:
                count2 = 0
            path_count_var: int = path_count_var + count1 * count2

        return path_count_var

    def path_count2b(self, f1, f2):
        developer_count_1 = dict()
        developer_count_2 = dict()
        path_count_var = 0

        if f1 in self.file_developer2b.keys():
            for developer in self.file_developer2b[f1]:
                if developer not in developer_count_1.keys():
                    developer_count_1[developer] = 1
                else:
                    developer_count_1[developer] = developer_count_1[developer] + 1
        if f2 in self.file_developer2b.keys():
            for developer in self.file_developer2b[f2]:
                if developer not in developer_count_2.keys():
                    developer_count_2[developer] = 1
                else:
                    developer_count_2[developer] = developer_count_2[developer] + 1

        for key in developer_count_1.keys():
            count1 = developer_count_1[key]
            if key in developer_count_2.keys():
                count2 = developer_count_2[key]
            else:
                count2 = 0
            path_count_var: int = path_count_var + count1 * count2

        return path_count_var

    def path_count2a(self, f1, f2):
        developer_count_1 = dict()
        developer_count_2 = dict()
        path_count_var = 0

        if f1 in self.file_developer_2a.keys():
            for developer in self.file_developer_2a[f1]:
                if developer not in developer_count_1.keys():
                    developer_count_1[developer] = 1
                else:
                    developer_count_1[developer] = developer_count_1[developer] + 1
        if f2 in self.file_developer_2a.keys():
            for developer in self.file_developer_2a[f2]:
                if developer not in developer_count_2.keys():
                    developer_count_2[developer] = 1
                else:
                    developer_count_2[developer] = developer_count_2[developer] + 1

        for key in developer_count_1.keys():
            count1 = developer_count_1[key]
            if key in developer_count_2.keys():
                count2 = developer_count_2[key]
            else:
                count2 = 0
            path_count_var: int = path_count_var + count1 * count2

        return path_count_var

    def path_count1(self, f1, f2):
        issue_count_1 = dict()
        issue_count_2 = dict()
        path_count_var = 0

        if f1 in self.file_issue.keys():
            for issue in self.file_issue[f1]:
                if issue not in issue_count_1.keys():
                    issue_count_1[issue] = 1
                else:
                    issue_count_1[issue] = issue_count_1[issue] + 1
        if f2 in self.file_issue.keys():
            for issue in self.file_issue[f2]:
                if issue not in issue_count_2.keys():
                    issue_count_2[issue] = 1
                else:
                    issue_count_2[issue] = issue_count_2[issue] + 1

        for key in issue_count_1.keys():
            count1 = issue_count_1[key]
            if key in issue_count_2.keys():
                count2 = issue_count_2[key]
            else:
                count2 = 0
            path_count_var: int = path_count_var + count1 * count2

        return path_count_var

    def build_commit_set_of_files(self):
        for element in self.commit_file_records:
            file_id = element["file_id"]
            commit_id = element["commit_id"]
            if commit_id not in self.commit_set_of_files.keys():
                self.commit_set_of_files[commit_id] = set()
            self.commit_set_of_files[commit_id].add(file_id)

    def build_commit_set_of_src_files(self, project):
        self.commit_set_of_src_files.clear()
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

    def build_commit_time(self):
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

    def build_file_commits(self):
        for element in self.commit_file_records:
            file_id = element["file_id"]
            commit_id = element["commit_id"]
            if file_id in self.file_path_data.keys():
                path1 = self.file_path_data[file_id]
                pos1 = path1.rfind('/')
                if pos1 != -1:
                    trim1 = path1[0:pos1:1]
                    modules1 = trim1.split("/")
                    if "src" in modules1:
                        if file_id not in self.file_commits_data.keys():
                            self.file_commits_data[file_id] = list()
                        if commit_id not in self.file_commits_data[file_id]:
                            self.file_commits_data[file_id].append(commit_id)
        print(f"Length of file_commits = {len(self.file_commits_data)}")

    def build_commit_issue(self):
        for element in self.commit_issue_records:
            try:
                issue_id = element["linked_issue_ids"]
            except KeyError:
                issue_id = "0"
            if issue_id and issue_id != '0':
                commit_id = element["_id"]
                if commit_id not in self.commit_issue_data.keys():
                    self.commit_issue_data[commit_id] = list()
                if issue_id not in self.commit_issue_data[commit_id]:
                    self.commit_issue_data[commit_id] = issue_id

    def build_file_issue(self):
        for file in self.file_commits_data.keys():
            commit = self.file_commits_data[file]
            r = len(commit)
            for i in range(r):
                if commit[i] in self.commit_issue_data.keys():
                    issue = self.commit_issue_data[commit[i]]
                    if file not in self.file_issue.keys():
                        self.file_issue[file] = list()
                    if issue not in self.file_issue[file]:
                        self.file_issue[file].append(issue)
        for file in self.file_issue.keys():
            self.file_issue[file] = list(itertools.chain.from_iterable(self.file_issue[file]))
        print(f"Length of file_issue = {len(self.file_issue)}")

    def build_commit_author(self):
        for element in self.commit_project_records:
            commit_id = element["_id"]
            author_id = self.BRID.reverse_identity_dict[element["author_id"]]
            if commit_id not in self.commit_author_data.keys():
                self.commit_author_data[commit_id] = list()
            if author_id not in self.commit_author_data[commit_id]:
                self.commit_author_data[commit_id].append(author_id)

    def build_file_developer_2a(self):
        for file in self.file_commits_data.keys():
            commit = self.file_commits_data[file]
            r = len(commit)
            for i in range(r):
                if commit[i] in self.commit_author_data.keys():
                    author = self.commit_author_data[commit[i]]
                    if file not in self.file_developer_2a.keys():
                        self.file_developer_2a[file] = list()
                    if author not in self.file_developer_2a[file]:
                        self.file_developer_2a[file].append(author)
        for file in self.file_developer_2a.keys():
            self.file_developer_2a[file] = list(itertools.chain.from_iterable(self.file_developer_2a[file]))

    def build_commit_committer(self):
        for element in self.commit_project_records:
            commit_id = element["_id"]
            committer_id = self.BRID.reverse_identity_dict[element["committer_id"]]
            if commit_id not in self.commit_committer_data.keys():
                self.commit_committer_data[commit_id] = list()
            if committer_id not in self.commit_committer_data[commit_id]:
                self.commit_committer_data[commit_id].append(committer_id)

    def build_file_developer2b(self):
        for file in self.file_commits_data.keys():
            commit = self.file_commits_data[file]
            r = len(commit)
            for i in range(r):
                if commit[i] in self.commit_committer_data.keys():
                    committer = self.commit_committer_data[commit[i]]
                    if file not in self.file_developer2b.keys():
                        self.file_developer2b[file] = list()
                    if committer not in self.file_developer2b[file]:
                        self.file_developer2b[file].append(committer)
        for file in self.file_developer2b.keys():
            self.file_developer2b[file] = list(itertools.chain.from_iterable(self.file_developer2b[file]))

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
                    if file not in self.file_level1.keys():
                        self.file_level1[file] = list()
                    if modules[sz - 1] not in self.file_level1[file]:
                        self.file_level1[file].append(modules[sz - 1])

    def build_file_level2(self):
        for file in self.file_path_data.keys():
            path = self.file_path_data[file]
            pos = path.rfind('/')
            if pos != -1:
                trim = path[0:pos:1]
                modules = trim.split("/")
                sz = len(modules)
                if sz >= 2:
                    if file not in self.file_level2.keys():
                        self.file_level2[file] = list()
                    if modules[sz - 2] not in self.file_level2[file]:
                        self.file_level2[file].append(modules[sz - 2])

    def build_file_level3(self):
        for file in self.file_path_data.keys():
            path = self.file_path_data[file]
            pos = path.rfind('/')
            if pos != -1:
                trim = path[0:pos:1]
                modules = trim.split("/")
                sz = len(modules)
                if sz >= 3:
                    if file not in self.file_level3.keys():
                        self.file_level3[file] = list()
                    if modules[sz - 3] not in self.file_level3[file]:
                        self.file_level3[file].append(modules[sz - 3])

    def build_commit_refactor(self):
        for element in self.commit_refactor_records:
            refactor_id = element["_id"]
            commit_id = element["commit_id"]
            if commit_id not in self.commit_refactor_data.keys():
                self.commit_refactor_data[commit_id] = list()
            if refactor_id not in self.commit_refactor_data[commit_id]:
                self.commit_refactor_data[commit_id].append(refactor_id)

    def build_file_refactor(self):
        for file in self.file_commits_data.keys():
            commit = self.file_commits_data[file]
            r = len(commit)
            for i in range(r):
                if commit[i] in self.commit_refactor_data.keys():
                    refactor = self.commit_refactor_data[commit[i]]
                    if file not in self.file_developer4.keys():
                        self.file_developer4[file] = list()
                    if refactor not in self.file_developer4[file]:
                        self.file_developer4[file].append(refactor)
        for file in self.file_developer4.keys():
            self.file_developer4[file] = list(itertools.chain.from_iterable(self.file_developer4[file]))

    def build_file_developer5(self):
        for file in self.file_commits_data.keys():
            commit = self.file_commits_data[file]
            r = len(commit)
            for i in range(r):
                if commit[i] in self.commit_author_data.keys():
                    author = self.commit_author_data[commit[i]]
                    if file not in self.file_developer5.keys():
                        self.file_developer5[file] = list()
                    if author not in self.file_developer5[file]:
                        self.file_developer5[file].append(author)
        for file in self.file_developer5.keys():
            self.file_developer5[file] = list(itertools.chain.from_iterable(self.file_developer5[file]))

    def build_developer5_issue(self):
        for element in self.issue_author_records:
            developer_id = self.BRID.reverse_identity_dict[element["author_id"]]
            issue_id = element["issue_id"]
            if developer_id not in self.developer5_issue.keys():
                self.developer5_issue[developer_id] = list()
            if issue_id not in self.developer5_issue[developer_id]:
                self.developer5_issue[developer_id].append(issue_id)

    def file_issue5_mapping(self):
        for file in self.file_developer5.keys():
            for developer in self.file_developer5[file]:
                if (developer in self.developer5_issue.keys()) and (self.developer5_issue[developer]):
                    for issue in self.developer5_issue[developer]:
                        if file not in self.file_issue5.keys():
                            self.file_issue5[file] = list()
                            self.file_issue5_unique[file] = list()
                        if issue not in self.file_issue5_unique[file]:
                            self.file_issue5_unique[file].append(issue)
                        self.file_issue5[file].append(issue)

    def build_issue_developer(self):
        for element in self.issue_author_records:
            developer_id = self.BRID.reverse_identity_dict[element["author_id"]]
            issue_id = element["issue_id"]
            if issue_id not in self.issue_developer.keys():
                self.issue_developer[issue_id] = list()
            if developer_id not in self.issue_developer[issue_id]:
                self.issue_developer[issue_id].append(developer_id)
        print(f"Length of developer_issue = {len(self.issue_developer)}")

    def file_developer_mapping(self):
        for file in self.file_issue.keys():
            for issue in self.file_issue[file]:
                if (issue in self.issue_developer.keys()) and (self.issue_developer[issue]):
                    for developer in self.issue_developer[issue]:
                        if file not in self.file_developer6.keys():
                            self.file_developer6[file] = list()
                            self.file_developer6_unique[file] = list()
                        if developer not in self.file_developer6[file]:
                            self.file_developer6_unique[file].append(developer)
                        self.file_developer6[file].append(developer)

    def build_issue_assignee(self):
        for element in self.issue_assignee_records:
            try:
                assignee_id = self.BRID.reverse_identity_dict[element["assignee_id"]]
            except KeyError:
                assignee_id = "0"
            if assignee_id and assignee_id != '0':
                issue_id = element["_id"]
                if issue_id not in self.issue_assignee.keys():
                    self.issue_assignee[issue_id] = list()
                if assignee_id not in self.issue_assignee[issue_id]:
                    self.issue_assignee[issue_id].append(assignee_id)

    def build_file_assignee(self):
        for file in self.file_issue.keys():
            issue = self.file_issue[file]
            r = len(issue)
            for i in range(r):
                if issue[i] in self.issue_assignee.keys():
                    assignee = self.issue_assignee[issue[i]]
                    if file not in self.file_assignee.keys():
                        self.file_assignee[file] = list()
                        self.file_assignee_unique[file] = list()
                    if assignee not in self.file_assignee_unique[file]:
                        self.file_assignee_unique[file].append(assignee)
                    self.file_assignee[file].append(assignee)
        for file in self.file_assignee:
            self.file_assignee[file] = list(itertools.chain.from_iterable(self.file_assignee[file]))

    def build_file_assignee_mapping(self):
        for file in self.file_issue.keys():
            for issue in self.file_issue[file]:
                if (issue in self.issue_assignee.keys()) and (self.issue_assignee[issue]):
                    for assignee in self.issue_assignee[issue]:
                        if file not in self.file_assignee_mapping.keys():
                            self.file_assignee_mapping[file] = list()
                            self.file_assignee_mapping_unique[file] = list()
                        if assignee not in self.file_assignee_mapping[file]:
                            self.file_assignee_mapping_unique[file].append(assignee)
                        self.file_assignee_mapping[file].append(assignee)

    def build_issue_developer8(self):
        for element in self.issue_author_records:
            developer_id = self.BRID.reverse_identity_dict[element["author_id"]]
            issue_id = element["issue_id"]
            if issue_id not in self.issue_developer8.keys():
                self.issue_developer8[issue_id] = list()
            if developer_id not in self.issue_developer8[issue_id]:
                self.issue_developer8[issue_id].append(developer_id)

    def build_file_developer8(self):
        for file in self.file_issue.keys():
            issue = self.file_issue[file]
            r = len(issue)
            for i in range(r):
                if issue[i] in self.issue_developer8.keys():
                    developer = self.issue_developer8[issue[i]]
                    if file not in self.file_developer8.keys():
                        self.file_developer8[file] = list()
                        self.file_developer8_unique[file] = list()
                    if developer not in self.file_developer8_unique[file]:
                        self.file_developer8_unique[file].append(developer)
                    self.file_developer8[file].append(developer)
        for file in self.file_developer8.keys():
            self.file_developer8[file] = list(itertools.chain.from_iterable(self.file_developer8[file]))

    def build_file_developer8_mapping(self):
        for file in self.file_issue.keys():
            for issue in self.file_issue[file]:
                if (issue in self.issue_developer8.keys()) and (self.issue_developer8[issue]):
                    for developer in self.issue_developer8[issue]:
                        if file not in self.file_developer8_mapping.keys():
                            self.file_developer8_mapping[file] = list()
                            self.file_developer8_mapping_unique[file] = list()
                        if developer not in self.file_developer8_mapping_unique[file]:
                            self.file_developer8_mapping_unique[file].append(developer)
                        self.file_developer8_mapping[file].append(developer)

    def build_issue_component(self):
        for element in self.issue_component_records:
            try:
                component_id = element["components"]
            except KeyError:
                component_id = "0"
            if component_id and component_id != '0':
                issue_id = element["_id"]
                if issue_id not in self.issue_component.keys():
                    self.issue_component[issue_id] = list()
                if component_id not in self.issue_component[issue_id]:
                    self.issue_component[issue_id].append(component_id)
        for issue in self.issue_component:
            self.issue_component[issue] = list(itertools.chain.from_iterable(self.issue_component[issue]))

    def file_component_mapping(self):
        for file in self.file_issue.keys():
            for issue in self.file_issue[file]:
                if (issue in self.issue_component.keys()) and (self.issue_component[issue]):
                    for component in self.issue_component[issue]:
                        if file not in self.file_component.keys():
                            self.file_component[file] = list()
                            self.file_component_unique[file] = list()
                        if component not in self.file_component_unique[file]:
                            self.file_component_unique[file].append(component)
                        self.file_component[file].append(component)

    def file_developer11_mapping(self):
        for file in self.file_issue.keys():
            for issue in self.file_issue[file]:
                if (issue in self.issue_assignee.keys()) and (self.issue_assignee[issue]):
                    for developer in self.issue_assignee[issue]:
                        if file not in self.file_developer11.keys():
                            self.file_developer11[file] = list()
                            self.file_developer11_unique[file] = list()
                        if developer not in self.file_developer11_unique[file]:
                            self.file_developer11_unique[file].append(developer)
                        self.file_developer11[file].append(developer)

    def build_issue_reporter(self):
        for element in self.issue_reporter_records:
            try:
                reporter_id = self.BRID.reverse_identity_dict[element["reporter_id"]]
            except KeyError:
                reporter_id = "0"
            if reporter_id and reporter_id != '0':
                issue_id = element["_id"]
                if issue_id not in self.issue_reporter.keys():
                    self.issue_reporter[issue_id] = list()
                if reporter_id not in self.issue_reporter[issue_id]:
                    self.issue_reporter[issue_id].append(reporter_id)

    def file_developer12_mapping(self):
        for file in self.file_issue.keys():
            for issue in self.file_issue[file]:
                if (issue in self.issue_reporter.keys()) and (self.issue_reporter[issue]):
                    for developer in self.issue_reporter[issue]:
                        if file not in self.file_developer12.keys():
                            self.file_developer12[file] = list()
                            self.file_developer12_unique[file] = list()
                        if developer not in self.file_developer12_unique[file]:
                            self.file_developer12_unique[file].append(developer)
                        self.file_developer12[file].append(developer)

    def print_data(self, project):
        c = 0
        time_list = list()
        for commit in self.commit_time:
            if self.commit_project_data[commit] == project:
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
        # t1 = t1min1 + relativedelta(years=2)
        # t2 = t1 + relativedelta(months=6)
        # t3 = t1 + relativedelta(months=12)
        # t4 = t1 + relativedelta(months=18)
        # t5 = t1 + relativedelta(months=24)
        print(f"t1 = {t1}")
        print(f"t2 = {t2}")
        print(f"t3 = {t3}")
        print(f"t4 = {t4}")
        print(f"t5 = {t5}")
        print(f"t6 = {t6}")
        print(f"t7 = {t7}")
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
        self.build_training_data(t1, t2, t3, t4, project)

    def build_training_data(self, t1, t2, t3, t4, project):
        self.training_data1.clear()
        self.training_data2.clear()
        for commit in self.commit_time:
            if self.commit_project_data[commit] == project:
                if t1 <= self.commit_time[commit] <= t2:
                    self.training_data1[commit] = self.commit_time[commit]
                if t3 <= self.commit_time[commit] <= t4:
                    self.training_data2[commit] = self.commit_time[commit]

    def build_training_file1(self, project):
        self.g1.clear()
        for commit in self.commit_set_of_src_files:
            if self.commit_project_data[commit] == project:
                if commit in self.training_data1:
                    files = self.commit_set_of_src_files[commit]
                    # if 30 >= len(files) > 0:
                    ordered_pairs = {(x, y) for x in files for y in files if x != y}
                    self.g1.add_edges_from(ordered_pairs)
        print(f"Length of edges training data 1 = {len(self.g1.edges())}")
        print(f"Length of nodes training data 1 = {len(self.g1.nodes())}")

    def build_training_file2(self, project):
        self.g2.clear()
        for commit in self.commit_set_of_src_files:
            if self.commit_project_data[commit] == project:
                if commit in self.training_data2:
                    files = self.commit_set_of_src_files[commit]
                    # if 30 >= len(files) >= 1:
                    ordered_pairs = {(x, y) for x in files for y in files if x != y}
                    self.g2.add_edges_from(ordered_pairs)
        print(f"Length of edges training data 2 = {len(self.g2.edges())}")
        print(f"Length of nodes training data 2 = {len(self.g2.nodes())}")

    def compare_graph(self, project):
        cd = 0
        self.s.clear()
        self.s1.clear()
        self.s2.clear()
        self.s3.clear()
        unique_pairs = list()
        r = nx.intersection(self.g1, self.g2)
        print(f"Length of common nodes = {len(r.nodes())}")
        print(f"Length of common edges = {len(r.edges())}")
        for i in combinations(r.nodes(), 2):
            cd = cd + 1
            unique_pairs.append(i)
        print(f"unique_pairs length = {len(unique_pairs)}")
        for a, b in unique_pairs:
            d = (a, b)
            if not self.g1.has_edge(a, b):
                self.s.append(d)
        print(f"s length = {len(self.s)}")
        for a, b in self.s:
            d = (a, b)
            if self.g2.has_edge(a, b):
                self.s1.append(d)
            else:
                self.s2.append(d)
        ls1 = len(self.s1)
        print(f"s1 length = {ls1}")
        ls2 = len(self.s2)
        print(f"s2 length = {ls2}")
        for a, b in self.s1:
            d = (a, b)
            self.s3.append(d)
        for a, b in self.s2:
            d = (a, b)
            self.s3.append(d)
        ls3 = len(self.s3)
        print(f"s3 length = {ls3}")
        unique_pairs.clear()
        r.clear()
        self.spawn_link_prediction_pool(project)

    def build_link_prediction(self, project):
        z1 = set()
        z2 = set()
        print(f"begin neighbors {project}")
        c = 0
        for a, b in self.s3:
            c = c + 1
            z1.clear()
            z2.clear()
            d = (a, b)
            z1 = set(self.g1.neighbors(a)).intersection(set(self.g1.neighbors(b)))
            self.link_common_neighbors[d] = len(z1)
            breakpoint()
            z2 = set(self.g1.neighbors(a)).union(set(self.g1.neighbors(b)))
            self.link_common_neighbors_union[d] = len(z2)
            self.link_jc[d] = len(z1) / len(z2)
            paa = (nx.adamic_adar_index(self.g1, [d]))
            for pa in paa:
                self.link_aa[d] = pa[2]
            self.link_pa[d] = len(set(self.g1.neighbors(a))) * len(set(self.g1.neighbors(b)))
            if nx.has_path(self.g1, source=a, target=b):
                self.link_shortest_path[d] = nx.shortest_path_length(self.g1, source=a, target=b)
            else:
                self.link_shortest_path[d] = 0
            y1 = self.path_count1(a, b)
            self.link_f_i_f[d] = y1
            y2a = self.path_count2a(a, b)
            self.link_f_a_f[d] = y2a
            y2b = self.path_count2b(a, b)
            self.link_f_c_f[d] = y2b
            y3a = self.path_count3a(a, b)
            self.link_f_l1_f[d] = y3a
            y3b = self.path_count3b(a, b)
            self.link_f_l2_f[d] = y3b
            y3c = self.path_count3c(a, b)
            self.link_f_l3_f[d] = y3c
            y4 = self.path_count4(a, b)
            self.link_f_r_f[d] = y4
            y6 = self.path_count6(a, b)
            self.link_f_i_d_i_f[d] = y6
            y7 = self.path_count7(a, b)
            self.link_f_d_i_f[d] = y7
            y8 = self.path_count8(a, b)
            self.link_f_a_i_f[d] = y8
            y9 = self.path_count9(a, b)
            self.link_f_i_c_i_f[d] = y9
            y11 = self.path_count11(a, b)
            self.link_f_i_a_i_f[d] = y11
            y12 = self.path_count12(a, b)
            self.link_f_i_r_i_f[d] = y12
            if self.g2.has_edge(a, b):
                self.link_class_variable[d] = 1
            else:
                self.link_class_variable[d] = 0

        csv_file = f"F://{project}_no_fatty_commits_training.csv"
        result_file = open(csv_file, "w")
        result_file.write("ID-a")
        result_file.write(",")
        result_file.write("ID-b")
        result_file.write(",")
        result_file.write("CN")
        result_file.write(",")
        result_file.write("TN")
        result_file.write(",")
        result_file.write("JC")
        result_file.write(",")
        result_file.write("AA")
        result_file.write(",")
        result_file.write("PA")
        result_file.write(",")
        result_file.write("SPL")
        result_file.write(",")
        result_file.write("F-I-F")
        result_file.write(",")
        result_file.write("F-A-F")
        result_file.write(",")
        result_file.write("F-C-F")
        result_file.write(",")
        result_file.write("F-M1-F")
        result_file.write(",")
        result_file.write("F-M2-F")
        result_file.write(",")
        result_file.write("F-M3-F")
        result_file.write(",")
        result_file.write("F-R-F")
        result_file.write(",")
        result_file.write("F-I-D-I-F")
        result_file.write(",")
        result_file.write("F-D-I-F")
        result_file.write(",")
        result_file.write("F-A-I-F")
        result_file.write(",")
        result_file.write("F-I-C-I-F")
        result_file.write(",")
        result_file.write("F-I-A-I-F")
        result_file.write(",")
        result_file.write("F-I-R-I-F")
        result_file.write(",")
        result_file.write("CV")
        result_file.write(",")
        result_file.write("\n")
        for e in self.link_common_neighbors:
            result_file.write(str(e))
            result_file.write(",")
            result_file.write(str(self.link_common_neighbors[e]))
            result_file.write(",")
            result_file.write(str(self.link_common_neighbors_union[e]))
            result_file.write(",")
            result_file.write(str(self.link_jc[e]))
            result_file.write(",")
            result_file.write(str(self.link_aa[e]))
            result_file.write(",")
            result_file.write(str(self.link_pa[e]))
            result_file.write(",")
            result_file.write(str(self.link_shortest_path[e]))
            result_file.write(",")
            result_file.write(str(self.link_f_i_f[e]))
            result_file.write(",")
            result_file.write(str(self.link_f_a_f[e]))
            result_file.write(",")
            result_file.write(str(self.link_f_c_f[e]))
            result_file.write(",")
            result_file.write(str(self.link_f_l1_f[e]))
            result_file.write(",")
            result_file.write(str(self.link_f_l2_f[e]))
            result_file.write(",")
            result_file.write(str(self.link_f_l3_f[e]))
            result_file.write(",")
            result_file.write(str(self.link_f_r_f[e]))
            result_file.write(",")
            result_file.write(str(self.link_f_i_d_i_f[e]))
            result_file.write(",")
            result_file.write(str(self.link_f_d_i_f[e]))
            result_file.write(",")
            result_file.write(str(self.link_f_a_i_f[e]))
            result_file.write(",")
            result_file.write(str(self.link_f_i_c_i_f[e]))
            result_file.write(",")
            result_file.write(str(self.link_f_i_a_i_f[e]))
            result_file.write(",")
            result_file.write(str(self.link_f_i_r_i_f[e]))
            result_file.write(",")
            result_file.write(str(self.link_class_variable[e]))
            result_file.write(",")
            result_file.write("\n")
        result_file.close()
        del csv_file
        del result_file
        self.link_common_neighbors.clear()
        self.link_common_neighbors_union.clear()
        self.link_jc.clear()
        self.link_aa.clear()
        self.link_pa.clear()
        self.link_shortest_path.clear()
        self.link_f_i_f.clear()
        self.link_f_a_f.clear()
        self.link_f_c_f.clear()
        self.link_f_l1_f.clear()
        self.link_f_l2_f.clear()
        self.link_f_l3_f.clear()
        self.link_f_r_f.clear()
        self.link_f_i_d_i_f.clear()
        self.link_f_d_i_f.clear()
        self.link_f_a_i_f.clear()
        self.link_f_i_c_i_f.clear()
        self.link_f_i_a_i_f.clear()
        self.link_f_i_r_i_f.clear()
        self.link_class_variable.clear()
        print("\n\n---------------------------------------------------------------")

    def write_header(self, csv_file):
        # with csv_file:
        writer = csv.writer(csv_file)
        header = [
            "ID-a",
            "ID-b",
            "CN",
            "TN",
            "JC",
            "AA",
            "PA",
            "SPL",
            "F-I-F",
            "F-A-F",
            "F-C-F",
            "F-M1-F",
            "F-M2-F",
            "F-M3-F",
            "F-R-F",
            "F-I-D-I-F",
            "F-D-I-F",
            "F-A-I-F",
            "F-I-C-I-F",
            "F-I-A-I-F",
            "F-I-R-I-F",
            "CV"]
        writer.writerow(header)

    def pair_link_prediction(self, pair):
        project = pair[2]
        a, b = pair[0], pair[1]
        z1 = set()
        z2 = set()
        d = (a, b)
        z1 = set(self.g1.neighbors(a)).intersection(set(self.g1.neighbors(b)))
        self.link_common_neighbors[d] = len(z1)
        breakpoint()
        z2 = set(self.g1.neighbors(a)).union(set(self.g1.neighbors(b)))
        self.link_common_neighbors_union[d] = len(z2)
        self.link_jc[d] = len(z1) / len(z2)
        paa = (nx.adamic_adar_index(self.g1, [d]))
        for pa in paa:
            self.link_aa[d] = pa[2]
        self.link_pa[d] = len(set(self.g1.neighbors(a))) * len(set(self.g1.neighbors(b)))
        if nx.has_path(self.g1, source=a, target=b):
            self.link_shortest_path[d] = nx.shortest_path_length(self.g1, source=a, target=b)
        else:
            self.link_shortest_path[d] = 0
        y1 = self.path_count1(a, b)
        self.link_f_i_f[d] = y1
        y2a = self.path_count2a(a, b)
        self.link_f_a_f[d] = y2a
        y2b = self.path_count2b(a, b)
        self.link_f_c_f[d] = y2b
        y3a = self.path_count3a(a, b)
        self.link_f_l1_f[d] = y3a
        y3b = self.path_count3b(a, b)
        self.link_f_l2_f[d] = y3b
        y3c = self.path_count3c(a, b)
        self.link_f_l3_f[d] = y3c
        y4 = self.path_count4(a, b)
        self.link_f_r_f[d] = y4
        y6 = self.path_count6(a, b)
        self.link_f_i_d_i_f[d] = y6
        y7 = self.path_count7(a, b)
        self.link_f_d_i_f[d] = y7
        y8 = self.path_count8(a, b)
        self.link_f_a_i_f[d] = y8
        y9 = self.path_count9(a, b)
        self.link_f_i_c_i_f[d] = y9
        y11 = self.path_count11(a, b)
        self.link_f_i_a_i_f[d] = y11
        y12 = self.path_count12(a, b)
        self.link_f_i_r_i_f[d] = y12
        if self.g2.has_edge(a, b):
            self.link_class_variable[d] = 1
        else:
            self.link_class_variable[d] = 0
        # write to file
        csv_file = f"F://{project}_no_fatty_commits_training.csv"
        file = open(csv_file, "w")
        with file:
            writer = csv.writer(file)
            self.write_header(file)

        with file:
            writer = csv.writer(file)
            for e in self.link_common_neighbors:
                row = [str(e),
                       str(self.link_common_neighbors[e]),
                       str(self.link_common_neighbors_union[e]),
                       str(self.link_jc[e]),
                       str(self.link_aa[e]),
                       str(self.link_pa[e]),
                       str(self.link_shortest_path[e]),
                       str(self.link_f_i_f[e]),
                       str(self.link_f_a_f[e]),
                       str(self.link_f_c_f[e]),
                       str(self.link_f_l1_f[e]),
                       str(self.link_f_l2_f[e]),
                       str(self.link_f_l3_f[e]),
                       str(self.link_f_r_f[e]),
                       str(self.link_f_i_d_i_f[e]),
                       str(self.link_f_d_i_f[e]),
                       str(self.link_f_a_i_f[e]),
                       str(self.link_f_i_c_i_f[e]),
                       str(self.link_f_i_a_i_f[e]),
                       str(self.link_f_i_r_i_f[e]),
                       str(self.link_class_variable[e])
                       ]
                writer.writerows(row)

    def spawn_link_prediction_pool(self, project):
        # breakpoint()
        print(f"starting {project}")
        # args found
        args = [(ele[0], ele[1], project) for ele in self.s3]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self.pair_link_prediction, args)

        # result_file.close()
        # del csv_file
        # del result_file
        self.link_common_neighbors.clear()
        self.link_common_neighbors_union.clear()
        self.link_jc.clear()
        self.link_aa.clear()
        self.link_pa.clear()
        self.link_shortest_path.clear()
        self.link_f_i_f.clear()
        self.link_f_a_f.clear()
        self.link_f_c_f.clear()
        self.link_f_l1_f.clear()
        self.link_f_l2_f.clear()
        self.link_f_l3_f.clear()
        self.link_f_r_f.clear()
        self.link_f_i_d_i_f.clear()
        self.link_f_d_i_f.clear()
        self.link_f_a_i_f.clear()
        self.link_f_i_c_i_f.clear()
        self.link_f_i_a_i_f.clear()
        self.link_f_i_r_i_f.clear()
        self.link_class_variable.clear()
        print("\n\n---------------------------------------------------------------")


if __name__ == "__main__":
    st = datetime.now()
    print(f"start time : {datetime.now()}")
    t = Task2()
    print("\nfile set of paths building")
    t.build_file_path()
    print("\nCommit set of files")
    t.build_commit_set_of_files()
    print("\nCommit time building")
    t.build_commit_time()
    print("\nfiles set of commits building")
    t.build_file_commits()
    print("\ncommit set of issues building")
    t.build_commit_issue()
    print("\n1.file set of issues building")
    t.build_file_issue()
    print("\ncommit set of authors building")
    t.build_commit_author()
    print("\n2a.file set of authors building")
    t.build_file_developer_2a()
    print("\ncommit set of committers building")
    t.build_commit_committer()
    print("\n2b.file set of committers building")
    t.build_file_developer2b()
    print("\n3a.file level 1 mapping")
    t.build_file_level1()
    print("\n3b.file level 2 mapping")
    t.build_file_level2()
    print("\n3c.file level 3 mapping")
    t.build_file_level3()
    print("\ncommit set of refactor building")
    t.build_commit_refactor()
    print("\n4.file set of refactor building")
    t.build_file_refactor()
    print("\nissue set of commenter's building")
    t.build_issue_developer()
    print("\n6.file commenter mapping building")
    t.file_developer_mapping()
    print("\nissue set of assignees building")
    t.build_issue_assignee()
    print("\nfile set of assignees building")
    t.build_file_assignee()
    print("\n7.file assignee mapping through issue")
    t.build_file_assignee_mapping()
    print("\nissue set of authors building")
    t.build_issue_developer8()
    print("\nfile set of authors building")
    t.build_file_developer8()
    print("\n8.file author mapping through issue")
    t.build_file_developer8_mapping()
    print("\nissue set of components building")
    t.build_issue_component()
    print("\n9.file component mapping building")
    t.file_component_mapping()
    print("\n11.file assignee 11 mapping building")
    t.file_developer11_mapping()
    print("\nissue set of reporters building")
    t.build_issue_reporter()
    print("\n12.file reporter mapping building")
    t.file_developer12_mapping()
    print(f"build time : {datetime.now() - st}")
    # project_list = ['mahout', 'pdfbox', 'opennlp', 'openwebbeans', 'mina-sshd', 'freemarker',
    #                 'gora', 'helix', 'curator', 'flume',
    #                 'directory-fortress-core', 'storm', 'cxf-fediz', 'systemml',
    #                 'fineract', 'knox', 'streams', 'zeppelin', 'samza',
    #                 'directory-kerby', 'commons-rdf', 'nifi', 'eagle']
    # project_list =['xerces2-j',
    #                 'xmlgraphics-batik',
    #                 'commons-beanutils',
    #                 'commons-collections',
    #                 'commons-dbcp',
    #                 'commons-digester',
    #                 'jspwiki',
    #                 'santuario-java',
    #                 'commons-bcel',
    #                 'commons-validator',
    #                 'commons-io',
    #                 'commons-jcs',
    #                 'commons-jexl',
    #                 'commons-vfs',
    #                 'commons-lang',
    #                 'jena',
    #                 'commons-codec',
    #                 'commons-math',
    #                 'maven',
    #                 'commons-compress',
    #                 'commons-configuration',
    #                 'wss4j',
    #                 'derby',
    #                 'jackrabbit',
    #                 'nutch',
    #                 'httpcomponents-core',
    #                 'roller',
    #                 'ant-ivy',
    #                 'commons-scxml',
    #                 'archiva',
    #                 'activemq',
    #                 'httpcomponents-client',
    #                 'struts',
    #                 'openjpa',
    #                 'directory-studio',
    #                 'cayenne',
    #                 'tika',
    #                 'commons-imaging',
    #                 'zookeeper',
    #                 'mahout',
    #                 'pdfbox',
    #                 'opennlp',
    #                 'openwebbeans',
    #                 'mina-sshd',
    #                 'pig',
    #                 'manifoldcf',
    #                 'freemarker',
    #                 'gora',
    #                 'giraph',
    #                 'helix',
    #                 'curator',
    #                 'bigtop',
    #                 'kafka',
    #                 'flume',
    #                 'oozie',
    #                 'directory-fortress-core',
    #                 'storm',
    #                 'falcon',
    #                 'cxf-fediz',
    #                 'deltaspike',
    #                 'systemml',
    #                 'calcite',
    #                 'fineract',
    #                 'parquet-mr',
    #                 'knox',
    #                 'streams',
    #                 'tez',
    #                 'lens',
    #                 'zeppelin',
    #                 'samza',
    #                 'phoenix',
    #                 'directory-kerby',
    #                 'kylin',
    #                 'commons-rdf',
    #                 'ranger',
    #                 'nifi',
    #                 'eagle'
    #                 ]
    # project_list = ['pig', 'manifoldcf',
    #                 'giraph', 'bigtop', 'kafka', 'oozie',
    #                 'falcon', 'deltaspike', 'calcite',
    #                 'parquet-mr', 'tez', 'lens', 'phoenix',
    #                 'kylin', 'ranger']
    project_list = ['giraph']
    for p in project_list:
        print("\n\n---------------------------------------------------------------")
        print(f"{p}:")
        name_of_project = p
        t.build_commit_set_of_src_files(name_of_project)
        t.print_data(name_of_project)
        t.build_training_file1(name_of_project)
        t.build_training_file2(name_of_project)
        t.compare_graph(name_of_project)
    print(f"Start time : {st}")
    print(f"End time : {datetime.now()}")