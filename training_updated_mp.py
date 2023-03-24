import itertools
import multiprocessing

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
from multiprocessing import Process, Pool, get_context


def pair_link_prediction(chunk):
    # a, b, project, g1, g2 = pair[0], pair[1], pair[2], pair[3], pair[4]
    z1, z2 = set(), set()

    for a, b in chunk:
        z1.clear()
        z2.clear()
        d = (a, b)
        print(f"running pair = {d}")
        z1 = set(g1.neighbors(a)).intersection(
            set(g1.neighbors(b)))
        link_common_neighbors[d] = len(z1)
        breakpoint()
        z2 = set(g1.neighbors(a)).union(set(g1.neighbors(b)))
        link_common_neighbors_union[d] = len(z2)
        link_jc[d] = len(z1) / len(z2)
        paa = (nx.adamic_adar_index(g1, [d]))
        for pa in paa:
            link_aa[d] = pa[2]
        link_pa[d] = len(set(g1.neighbors(a))) * \
            len(set(g1.neighbors(b)))
        if nx.has_path(g1, source=a, target=b):
            link_shortest_path[d] = nx.shortest_path_length(
                g1, source=a, target=b)
        else:
            link_shortest_path[d] = 0
        y1 = path_count1(a, b)
        link_f_i_f[d] = y1
        y2a = path_count2a(a, b)
        link_f_a_f[d] = y2a
        y2b = path_count2b(a, b)
        link_f_c_f[d] = y2b
        y3a = path_count3a(a, b)
        link_f_l1_f[d] = y3a
        y3b = path_count3b(a, b)
        link_f_l2_f[d] = y3b
        y3c = path_count3c(a, b)
        link_f_l3_f[d] = y3c
        y4 = path_count4(a, b)
        link_f_r_f[d] = y4
        y6 = path_count6(a, b)
        link_f_i_d_i_f[d] = y6
        y7 = path_count7(a, b)
        link_f_d_i_f[d] = y7
        y8 = path_count8(a, b)
        link_f_a_i_f[d] = y8
        y9 = path_count9(a, b)
        link_f_i_c_i_f[d] = y9
        y11 = path_count11(a, b)
        link_f_i_a_i_f[d] = y11
        y12 = path_count12(a, b)
        link_f_i_r_i_f[d] = y12
        if g2.has_edge(a, b):
            link_class_variable[d] = 1
        else:
            link_class_variable[d] = 0


client = MongoClient("mongodb://localhost:27017/")
db = client["smartshark"]
commit_project = db["commit_with_project_info"]
commit_project_records = list(commit_project.find({}, {}))
commit_file = db["file_action"]
commit_refactoring = db["refactoring"]
issue_author = db["issue_comment"]
issue_reporter1 = db["issue"]
file_path = db["file"]
commit_issue_records = list(
    commit_project.find({}, {"_id": 1, "linked_issue_ids": 1}))
commit_file_records = list(
    commit_file.find({}, {"file_id": 1, "commit_id": 1}))
commit_refactor_records = list(
    commit_refactoring.find({}, {"_id": 1, "commit_id": 1}))
issue_author_records = list(
    issue_author.find({}, {"issue_id": 1, "author_id": 1}))
issue_reporter_records = list(
    issue_reporter1.find({}, {"_id": 1, "reporter_id": 1}))
issue_assignee_records = list(
    issue_reporter1.find({}, {"_id": 1, "assignee_id": 1}))
issue_component_records = list(
    issue_reporter1.find({}, {"_id": 1, "components": 1}))
file_path_records = list(
    file_path.find({}, {"_id": 1, "path": 1}))
client.close()
BRID = Build_reverse_identity_dictionary()
BRID.reading_identity_and_people_and_building_reverse_identity_dictionary()
commit_set_of_files = dict()
commit_project_data = dict()
commit_time_unsort = dict()
commit_time = dict()
training_data1 = dict()
training_data2 = dict()
g1 = nx.Graph()
g2 = nx.Graph()
s = list()
s1 = list()
s2 = list()
s3 = list()
link_common_neighbors = dict()
link_common_neighbors_union = dict()
link_jc = dict()
link_aa = dict()
link_pa = dict()
link_shortest_path = dict()
link_class_variable = dict()
commit_issue_data = dict()
file_commits_data = dict()
file_issue = dict()
link_f_i_f = dict()
commit_author_data = dict()
file_developer_2a = dict()
link_f_a_f = dict()
commit_committer_data = dict()
file_developer2b = dict()
link_f_c_f = dict()
file_path_data = dict()
file_level1 = dict()
link_f_l1_f = dict()
file_level2 = dict()
link_f_l2_f = dict()
file_level3 = dict()
link_f_l3_f = dict()
commit_refactor_data = dict()
file_developer4 = dict()
link_f_r_f = dict()
file_developer5 = dict()
developer5_issue = dict()
file_issue5 = dict()
file_issue5_unique = dict()
link_f_d_i_d_f = dict()
issue_developer = dict()
file_developer6 = dict()
file_developer6_unique = dict()
link_f_i_d_i_f = dict()
issue_assignee = dict()
file_assignee = dict()
file_assignee_unique = dict()
file_assignee_mapping = dict()
file_assignee_mapping_unique = dict()
link_f_d_i_f = dict()
issue_developer8 = dict()
file_developer8 = dict()
file_developer8_unique = dict()
file_developer8_mapping = dict()
file_developer8_mapping_unique = dict()
link_f_a_i_f = dict()
issue_component = dict()
file_component = dict()
file_component_unique = dict()
link_f_i_c_i_f = dict()
file_developer11 = dict()
file_developer11_unique = dict()
link_f_i_a_i_f = dict()
issue_reporter = dict()
file_developer12 = dict()
file_developer12_unique = dict()
link_f_i_r_i_f = dict()
commit_set_of_src_files = dict()


def path_count12(f1, f2):
    developer_count_1 = dict()
    developer_count_2 = dict()
    path_count_var = 0

    if f1 in file_developer12.keys():
        for developer in file_developer12[f1]:
            if developer not in developer_count_1.keys():
                developer_count_1[developer] = 1
            else:
                developer_count_1[developer] = developer_count_1[developer] + 1
    if f2 in file_developer12.keys():
        for developer in file_developer12[f2]:
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


def path_count11(f1, f2):
    developer_count_1 = dict()
    developer_count_2 = dict()
    path_count_var = 0

    if f1 in file_developer11.keys():
        for developer in file_developer11[f1]:
            if developer not in developer_count_1.keys():
                developer_count_1[developer] = 1
            else:
                developer_count_1[developer] = developer_count_1[developer] + 1
    if f2 in file_developer11.keys():
        for developer in file_developer11[f2]:
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


def path_count9(f1, f2):
    developer_count_1 = dict()
    developer_count_2 = dict()
    path_count_var = 0

    if f1 in file_component.keys():
        for developer in file_component[f1]:
            if developer not in developer_count_1.keys():
                developer_count_1[developer] = 1
            else:
                developer_count_1[developer] = developer_count_1[developer] + 1
    if f2 in file_component.keys():
        for developer in file_component[f2]:
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


def path_count8(f1, f2):
    developer_count = dict()
    developer_mapping_count = dict()
    path_count_var = 0

    if f1 in file_developer8.keys():
        for developer in file_developer8[f1]:
            if developer not in developer_count.keys():
                developer_count[developer] = 1
            else:
                developer_count[developer] = developer_count[developer] + 1

    if f2 in file_developer8_mapping.keys():
        for developer in file_developer8_mapping[f2]:
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


def path_count7(f1, f2):
    assignee_count = dict()
    assignee_mapping_count = dict()
    path_count_var = 0

    if f1 in file_assignee.keys():
        for assignee in file_assignee[f1]:
            if assignee not in assignee_count.keys():
                assignee_count[assignee] = 1
            else:
                assignee_count[assignee] = assignee_count[assignee] + 1

    if f2 in file_assignee_mapping.keys():
        for assignee in file_assignee_mapping[f2]:
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


def path_count6(f1, f2):
    developer_count_1 = dict()
    developer_count_2 = dict()
    path_count_var = 0

    if f1 in file_developer6.keys():
        for developer in file_developer6[f1]:
            if developer not in developer_count_1.keys():
                developer_count_1[developer] = 1
            else:
                developer_count_1[developer] = developer_count_1[developer] + 1
    if f2 in file_developer6.keys():
        for developer in file_developer6[f2]:
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


def path_count5(f1, f2):
    issue_count_1 = dict()
    issue_count_2 = dict()
    path_count_var = 0

    if f1 in file_issue5.keys():
        for issue in file_issue5[f1]:
            if issue not in issue_count_1.keys():
                issue_count_1[issue] = 1
            else:
                issue_count_1[issue] = issue_count_1[issue] + 1
    if f2 in file_issue5.keys():
        for issue in file_issue5[f2]:
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


def path_count4(f1, f2):
    developer_count_1 = dict()
    developer_count_2 = dict()
    path_count_var = 0

    if f1 in file_developer4.keys():
        for developer in file_developer4[f1]:
            if developer not in developer_count_1.keys():
                developer_count_1[developer] = 1
            else:
                developer_count_1[developer] = developer_count_1[developer] + 1
    if f2 in file_developer4.keys():
        for developer in file_developer4[f2]:
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


def path_count3c(f1, f2):
    level3_count1 = dict()
    level3_count2 = dict()
    path_count_var = 0

    if f1 in file_level3.keys():
        for level3 in file_level3[f1]:
            if level3 not in level3_count1.keys():
                level3_count1[level3] = 1
            else:
                level3_count1[level3] = level3_count1[level3] + 1

    if f2 in file_level3.keys():
        for level3 in file_level3[f2]:
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


def path_count3b(f1, f2):
    level2_count1 = dict()
    level2_count2 = dict()
    path_count_var = 0

    if f1 in file_level2.keys():
        for level2 in file_level2[f1]:
            if level2 not in level2_count1.keys():
                level2_count1[level2] = 1
            else:
                level2_count1[level2] = level2_count1[level2] + 1

    if f2 in file_level2.keys():
        for level2 in file_level2[f2]:
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


def path_count3a(f1, f2):
    level1_count1 = dict()
    level1_count2 = dict()
    path_count_var = 0

    if f1 in file_level1.keys():
        for level1 in file_level1[f1]:
            if level1 not in level1_count1.keys():
                level1_count1[level1] = 1
            else:
                level1_count1[level1] = level1_count1[level1] + 1

    if f2 in file_level1.keys():
        for level1 in file_level1[f2]:
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


def path_count2b(f1, f2):
    developer_count_1 = dict()
    developer_count_2 = dict()
    path_count_var = 0

    if f1 in file_developer2b.keys():
        for developer in file_developer2b[f1]:
            if developer not in developer_count_1.keys():
                developer_count_1[developer] = 1
            else:
                developer_count_1[developer] = developer_count_1[developer] + 1
    if f2 in file_developer2b.keys():
        for developer in file_developer2b[f2]:
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


def path_count2a(f1, f2):
    developer_count_1 = dict()
    developer_count_2 = dict()
    path_count_var = 0

    if f1 in file_developer_2a.keys():
        for developer in file_developer_2a[f1]:
            if developer not in developer_count_1.keys():
                developer_count_1[developer] = 1
            else:
                developer_count_1[developer] = developer_count_1[developer] + 1
    if f2 in file_developer_2a.keys():
        for developer in file_developer_2a[f2]:
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


def path_count1(f1, f2):
    issue_count_1 = dict()
    issue_count_2 = dict()
    path_count_var = 0

    if f1 in file_issue.keys():
        for issue in file_issue[f1]:
            if issue not in issue_count_1.keys():
                issue_count_1[issue] = 1
            else:
                issue_count_1[issue] = issue_count_1[issue] + 1
    if f2 in file_issue.keys():
        for issue in file_issue[f2]:
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


def build_commit_set_of_files():
    for element in commit_file_records:
        file_id = element["file_id"]
        commit_id = element["commit_id"]
        if commit_id not in commit_set_of_files.keys():
            commit_set_of_files[commit_id] = set()
        commit_set_of_files[commit_id].add(file_id)


def build_commit_set_of_src_files(project):
    commit_set_of_src_files.clear()
    for commit in commit_set_of_files:
        if commit_project_data[commit] == project:
            files = commit_set_of_files[commit]
            files = list(files)
            r = len(files)
            for i in range(r):
                path1 = file_path_data[files[i]]
                pos1 = path1.rfind('/')
                if pos1 != -1:
                    trim1 = path1[0:pos1:1]
                    modules1 = trim1.split("/")
                    if "src" in modules1:
                        if commit not in commit_set_of_src_files.keys():
                            commit_set_of_src_files[commit] = list()
                        if i not in commit_set_of_src_files[commit]:
                            commit_set_of_src_files[commit].append(
                                files[i])


def build_commit_time():
    for element in commit_project_records:
        commit_id = element["_id"]
        project_name = element["project_name_info"]["name"]
        c_time = element["committer_date"]
        if commit_id not in commit_project_data.keys():
            commit_project_data[commit_id] = project_name
        if commit_id not in commit_time_unsort.keys():
            commit_time_unsort[commit_id] = c_time
    commit_time = {key: value for key, value in sorted(commit_time_unsort.items(),
                                                       key=lambda item: item[1])}
    return commit_time


commit_time = build_commit_time()


def build_file_commits():
    for element in commit_file_records:
        file_id = element["file_id"]
        commit_id = element["commit_id"]
        if file_id in file_path_data.keys():
            path1 = file_path_data[file_id]
            pos1 = path1.rfind('/')
            if pos1 != -1:
                trim1 = path1[0:pos1:1]
                modules1 = trim1.split("/")
                if "src" in modules1:
                    if file_id not in file_commits_data.keys():
                        file_commits_data[file_id] = list()
                    if commit_id not in file_commits_data[file_id]:
                        file_commits_data[file_id].append(commit_id)
    print(f"Length of file_commits = {len(file_commits_data)}")


def build_commit_issue():
    for element in commit_issue_records:
        try:
            issue_id = element["linked_issue_ids"]
        except KeyError:
            issue_id = "0"
        if issue_id and issue_id != '0':
            commit_id = element["_id"]
            if commit_id not in commit_issue_data.keys():
                commit_issue_data[commit_id] = list()
            if issue_id not in commit_issue_data[commit_id]:
                commit_issue_data[commit_id] = issue_id


def build_file_issue():
    for file in file_commits_data.keys():
        commit = file_commits_data[file]
        r = len(commit)
        for i in range(r):
            if commit[i] in commit_issue_data.keys():
                issue = commit_issue_data[commit[i]]
                if file not in file_issue.keys():
                    file_issue[file] = list()
                if issue not in file_issue[file]:
                    file_issue[file].append(issue)
    for file in file_issue.keys():
        file_issue[file] = list(
            itertools.chain.from_iterable(file_issue[file]))
    print(f"Length of file_issue = {len(file_issue)}")


def build_commit_author():
    for element in commit_project_records:
        commit_id = element["_id"]
        author_id = BRID.reverse_identity_dict[element["author_id"]]
        if commit_id not in commit_author_data.keys():
            commit_author_data[commit_id] = list()
        if author_id not in commit_author_data[commit_id]:
            commit_author_data[commit_id].append(author_id)


def build_file_developer_2a():
    for file in file_commits_data.keys():
        commit = file_commits_data[file]
        r = len(commit)
        for i in range(r):
            if commit[i] in commit_author_data.keys():
                author = commit_author_data[commit[i]]
                if file not in file_developer_2a.keys():
                    file_developer_2a[file] = list()
                if author not in file_developer_2a[file]:
                    file_developer_2a[file].append(author)
    for file in file_developer_2a.keys():
        file_developer_2a[file] = list(
            itertools.chain.from_iterable(file_developer_2a[file]))


def build_commit_committer():
    for element in commit_project_records:
        commit_id = element["_id"]
        committer_id = BRID.reverse_identity_dict[element["committer_id"]]
        if commit_id not in commit_committer_data.keys():
            commit_committer_data[commit_id] = list()
        if committer_id not in commit_committer_data[commit_id]:
            commit_committer_data[commit_id].append(committer_id)


def build_file_developer2b():
    for file in file_commits_data.keys():
        commit = file_commits_data[file]
        r = len(commit)
        for i in range(r):
            if commit[i] in commit_committer_data.keys():
                committer = commit_committer_data[commit[i]]
                if file not in file_developer2b.keys():
                    file_developer2b[file] = list()
                if committer not in file_developer2b[file]:
                    file_developer2b[file].append(committer)
    for file in file_developer2b.keys():
        file_developer2b[file] = list(
            itertools.chain.from_iterable(file_developer2b[file]))


def build_file_path():
    for element in file_path_records:
        file_id = element["_id"]
        path = element["path"]
        if file_id not in file_path_data.keys():
            file_path_data[file_id] = path


def build_file_level1():
    for file in file_path_data.keys():
        path = file_path_data[file]
        pos = path.rfind('/')
        if pos != -1:
            trim = path[0:pos:1]
            modules = trim.split("/")
            sz = len(modules)
            if sz >= 1:
                if file not in file_level1.keys():
                    file_level1[file] = list()
                if modules[sz - 1] not in file_level1[file]:
                    file_level1[file].append(modules[sz - 1])


def build_file_level2():
    for file in file_path_data.keys():
        path = file_path_data[file]
        pos = path.rfind('/')
        if pos != -1:
            trim = path[0:pos:1]
            modules = trim.split("/")
            sz = len(modules)
            if sz >= 2:
                if file not in file_level2.keys():
                    file_level2[file] = list()
                if modules[sz - 2] not in file_level2[file]:
                    file_level2[file].append(modules[sz - 2])


def build_file_level3():
    for file in file_path_data.keys():
        path = file_path_data[file]
        pos = path.rfind('/')
        if pos != -1:
            trim = path[0:pos:1]
            modules = trim.split("/")
            sz = len(modules)
            if sz >= 3:
                if file not in file_level3.keys():
                    file_level3[file] = list()
                if modules[sz - 3] not in file_level3[file]:
                    file_level3[file].append(modules[sz - 3])


def build_commit_refactor():
    for element in commit_refactor_records:
        refactor_id = element["_id"]
        commit_id = element["commit_id"]
        if commit_id not in commit_refactor_data.keys():
            commit_refactor_data[commit_id] = list()
        if refactor_id not in commit_refactor_data[commit_id]:
            commit_refactor_data[commit_id].append(refactor_id)


def build_file_refactor():
    for file in file_commits_data.keys():
        commit = file_commits_data[file]
        r = len(commit)
        for i in range(r):
            if commit[i] in commit_refactor_data.keys():
                refactor = commit_refactor_data[commit[i]]
                if file not in file_developer4.keys():
                    file_developer4[file] = list()
                if refactor not in file_developer4[file]:
                    file_developer4[file].append(refactor)
    for file in file_developer4.keys():
        file_developer4[file] = list(
            itertools.chain.from_iterable(file_developer4[file]))


def build_file_developer5():
    for file in file_commits_data.keys():
        commit = file_commits_data[file]
        r = len(commit)
        for i in range(r):
            if commit[i] in commit_author_data.keys():
                author = commit_author_data[commit[i]]
                if file not in file_developer5.keys():
                    file_developer5[file] = list()
                if author not in file_developer5[file]:
                    file_developer5[file].append(author)
    for file in file_developer5.keys():
        file_developer5[file] = list(
            itertools.chain.from_iterable(file_developer5[file]))


def build_developer5_issue():
    for element in issue_author_records:
        developer_id = BRID.reverse_identity_dict[element["author_id"]]
        issue_id = element["issue_id"]
        if developer_id not in developer5_issue.keys():
            developer5_issue[developer_id] = list()
        if issue_id not in developer5_issue[developer_id]:
            developer5_issue[developer_id].append(issue_id)


def file_issue5_mapping():
    for file in file_developer5.keys():
        for developer in file_developer5[file]:
            if (developer in developer5_issue.keys()) and (developer5_issue[developer]):
                for issue in developer5_issue[developer]:
                    if file not in file_issue5.keys():
                        file_issue5[file] = list()
                        file_issue5_unique[file] = list()
                    if issue not in file_issue5_unique[file]:
                        file_issue5_unique[file].append(issue)
                    file_issue5[file].append(issue)


def build_issue_developer():
    for element in issue_author_records:
        developer_id = BRID.reverse_identity_dict[element["author_id"]]
        issue_id = element["issue_id"]
        if issue_id not in issue_developer.keys():
            issue_developer[issue_id] = list()
        if developer_id not in issue_developer[issue_id]:
            issue_developer[issue_id].append(developer_id)
    print(f"Length of developer_issue = {len(issue_developer)}")


def file_developer_mapping():
    for file in file_issue.keys():
        for issue in file_issue[file]:
            if (issue in issue_developer.keys()) and (issue_developer[issue]):
                for developer in issue_developer[issue]:
                    if file not in file_developer6.keys():
                        file_developer6[file] = list()
                        file_developer6_unique[file] = list()
                    if developer not in file_developer6[file]:
                        file_developer6_unique[file].append(developer)
                    file_developer6[file].append(developer)


def build_issue_assignee():
    for element in issue_assignee_records:
        try:
            assignee_id = BRID.reverse_identity_dict[element["assignee_id"]]
        except KeyError:
            assignee_id = "0"
        if assignee_id and assignee_id != '0':
            issue_id = element["_id"]
            if issue_id not in issue_assignee.keys():
                issue_assignee[issue_id] = list()
            if assignee_id not in issue_assignee[issue_id]:
                issue_assignee[issue_id].append(assignee_id)


def build_file_assignee():
    for file in file_issue.keys():
        issue = file_issue[file]
        r = len(issue)
        for i in range(r):
            if issue[i] in issue_assignee.keys():
                assignee = issue_assignee[issue[i]]
                if file not in file_assignee.keys():
                    file_assignee[file] = list()
                    file_assignee_unique[file] = list()
                if assignee not in file_assignee_unique[file]:
                    file_assignee_unique[file].append(assignee)
                file_assignee[file].append(assignee)
    for file in file_assignee:
        file_assignee[file] = list(
            itertools.chain.from_iterable(file_assignee[file]))


def build_file_assignee_mapping():
    for file in file_issue.keys():
        for issue in file_issue[file]:
            if (issue in issue_assignee.keys()) and (issue_assignee[issue]):
                for assignee in issue_assignee[issue]:
                    if file not in file_assignee_mapping.keys():
                        file_assignee_mapping[file] = list()
                        file_assignee_mapping_unique[file] = list()
                    if assignee not in file_assignee_mapping[file]:
                        file_assignee_mapping_unique[file].append(
                            assignee)
                    file_assignee_mapping[file].append(assignee)


def build_issue_developer8():
    for element in issue_author_records:
        developer_id = BRID.reverse_identity_dict[element["author_id"]]
        issue_id = element["issue_id"]
        if issue_id not in issue_developer8.keys():
            issue_developer8[issue_id] = list()
        if developer_id not in issue_developer8[issue_id]:
            issue_developer8[issue_id].append(developer_id)


def build_file_developer8():
    for file in file_issue.keys():
        issue = file_issue[file]
        r = len(issue)
        for i in range(r):
            if issue[i] in issue_developer8.keys():
                developer = issue_developer8[issue[i]]
                if file not in file_developer8.keys():
                    file_developer8[file] = list()
                    file_developer8_unique[file] = list()
                if developer not in file_developer8_unique[file]:
                    file_developer8_unique[file].append(developer)
                file_developer8[file].append(developer)
    for file in file_developer8.keys():
        file_developer8[file] = list(
            itertools.chain.from_iterable(file_developer8[file]))


def build_file_developer8_mapping():
    for file in file_issue.keys():
        for issue in file_issue[file]:
            if (issue in issue_developer8.keys()) and (issue_developer8[issue]):
                for developer in issue_developer8[issue]:
                    if file not in file_developer8_mapping.keys():
                        file_developer8_mapping[file] = list()
                        file_developer8_mapping_unique[file] = list()
                    if developer not in file_developer8_mapping_unique[file]:
                        file_developer8_mapping_unique[file].append(
                            developer)
                    file_developer8_mapping[file].append(developer)


def build_issue_component():
    for element in issue_component_records:
        try:
            component_id = element["components"]
        except KeyError:
            component_id = "0"
        if component_id and component_id != '0':
            issue_id = element["_id"]
            if issue_id not in issue_component.keys():
                issue_component[issue_id] = list()
            if component_id not in issue_component[issue_id]:
                issue_component[issue_id].append(component_id)
    for issue in issue_component:
        issue_component[issue] = list(
            itertools.chain.from_iterable(issue_component[issue]))


def file_component_mapping():
    for file in file_issue.keys():
        for issue in file_issue[file]:
            if (issue in issue_component.keys()) and (issue_component[issue]):
                for component in issue_component[issue]:
                    if file not in file_component.keys():
                        file_component[file] = list()
                        file_component_unique[file] = list()
                    if component not in file_component_unique[file]:
                        file_component_unique[file].append(component)
                    file_component[file].append(component)


def file_developer11_mapping():
    for file in file_issue.keys():
        for issue in file_issue[file]:
            if (issue in issue_assignee.keys()) and (issue_assignee[issue]):
                for developer in issue_assignee[issue]:
                    if file not in file_developer11.keys():
                        file_developer11[file] = list()
                        file_developer11_unique[file] = list()
                    if developer not in file_developer11_unique[file]:
                        file_developer11_unique[file].append(
                            developer)
                    file_developer11[file].append(developer)


def build_issue_reporter():
    for element in issue_reporter_records:
        try:
            reporter_id = BRID.reverse_identity_dict[element["reporter_id"]]
        except KeyError:
            reporter_id = "0"
        if reporter_id and reporter_id != '0':
            issue_id = element["_id"]
            if issue_id not in issue_reporter.keys():
                issue_reporter[issue_id] = list()
            if reporter_id not in issue_reporter[issue_id]:
                issue_reporter[issue_id].append(reporter_id)


def file_developer12_mapping():
    for file in file_issue.keys():
        for issue in file_issue[file]:
            if (issue in issue_reporter.keys()) and (issue_reporter[issue]):
                for developer in issue_reporter[issue]:
                    if file not in file_developer12.keys():
                        file_developer12[file] = list()
                        file_developer12_unique[file] = list()
                    if developer not in file_developer12_unique[file]:
                        file_developer12_unique[file].append(
                            developer)
                    file_developer12[file].append(developer)


def print_data(project):
    c = 0
    time_list = list()
    for commit in commit_time:
        # breakpoint()
        if commit_project_data[commit] == project:
            time_list.append(commit_time[commit])
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
    for commit in commit_time:
        if commit_project_data[commit] == project:
            if t1 <= commit_time[commit] <= t7:
                c = c + 1
    print(f"Number of commits between t1 and t7 = {c}")
    c2 = 0
    for commit in commit_time:
        if commit_project_data[commit] == project:
            if t1 <= commit_time[commit] <= t2:
                c2 = c2 + 1
    print(f"Number of commits between t1 and t2 : {c2}")
    c1 = 0
    for commit in commit_time:
        if commit_project_data[commit] == project:
            if t2 < commit_time[commit] <= t3:
                c1 = c1 + 1
    print(f"Number of commits between t2 and t3 : {c1}")
    c3 = 0
    for commit in commit_time:
        if commit_project_data[commit] == project:
            if t3 < commit_time[commit] <= t4:
                c3 = c3 + 1
    print(f"Number of commits between t3 and t4 : {c3}")
    c4 = 0
    for commit in commit_time:
        if commit_project_data[commit] == project:
            if t4 < commit_time[commit] <= t5:
                c4 = c4 + 1
    print(f"Number of commits between t4 and t5 : {c4}")
    c5 = 0
    for commit in commit_time:
        if commit_project_data[commit] == project:
            if t5 < commit_time[commit] <= t6:
                c5 = c5 + 1
    print(f"Number of commits between t5 and t6 : {c5}")
    c6 = 0
    for commit in commit_time:
        if commit_project_data[commit] == project:
            if t6 < commit_time[commit] <= t7:
                c6 = c6 + 1
    print(f"Number of commits between t6 and t7 : {c6}")
    build_training_data(t1, t2, t3, t4, project)


def build_training_data(t1, t2, t3, t4, project):
    training_data1.clear()
    training_data2.clear()
    for commit in commit_time:
        if commit_project_data[commit] == project:
            if t1 <= commit_time[commit] <= t2:
                training_data1[commit] = commit_time[commit]
            if t3 <= commit_time[commit] <= t4:
                training_data2[commit] = commit_time[commit]


def build_training_file1(project):
    g1.clear()
    for commit in commit_set_of_src_files:
        if commit_project_data[commit] == project:
            if commit in training_data1:
                files = commit_set_of_src_files[commit]
                # if 30 >= len(files) > 0:
                ordered_pairs = {(x, y)
                                 for x in files for y in files if x != y}
                g1.add_edges_from(ordered_pairs)
    print(f"Length of edges training data 1 = {len(g1.edges())}")
    print(f"Length of nodes training data 1 = {len(g1.nodes())}")


def build_training_file2(project):
    g2.clear()
    for commit in commit_set_of_src_files:
        if commit_project_data[commit] == project:
            if commit in training_data2:
                files = commit_set_of_src_files[commit]
                # if 30 >= len(files) >= 1:
                ordered_pairs = {(x, y)
                                 for x in files for y in files if x != y}
                g2.add_edges_from(ordered_pairs)
    print(f"Length of edges training data 2 = {len(g2.edges())}")
    print(f"Length of nodes training data 2 = {len(g2.nodes())}")


def compare_graph(project):
    cd = 0
    s.clear()
    s1.clear()
    s2.clear()
    s3.clear()
    unique_pairs = list()
    r = nx.intersection(g1, g2)
    print(f"Length of common nodes = {len(r.nodes())}")
    print(f"Length of common edges = {len(r.edges())}")
    for i in combinations(r.nodes(), 2):
        cd = cd + 1
        unique_pairs.append(i)
    print(f"unique_pairs length = {len(unique_pairs)}")
    for a, b in unique_pairs:
        d = (a, b)
        if not g1.has_edge(a, b):
            s.append(d)
    print(f"s length = {len(s)}")
    for a, b in s:
        d = (a, b)
        if g2.has_edge(a, b):
            s1.append(d)
        else:
            s2.append(d)
    ls1 = len(s1)
    print(f"s1 length = {ls1}")
    ls2 = len(s2)
    print(f"s2 length = {ls2}")
    for a, b in s1:
        d = (a, b)
        s3.append(d)
    for a, b in s2:
        d = (a, b)
        s3.append(d)
    ls3 = len(s3)
    print(f"s3 length = {ls3}")
    unique_pairs.clear()
    r.clear()
    # spawn_link_prediction_pool(project)
    build_link_prediction(project)


def build_link_prediction(project):
    z1 = set()
    z2 = set()
    print(f"begin neighbors {project}")
    c = 0
    with get_context("spawn").Pool(processes=8) as pool:
        args = s3
        chunks = []
        chunk_size = int(len(args)/8)
        chunk_indexes = list(range(0, len(args), chunk_size))
        for i, idx in enumerate(chunk_indexes):
            print(f"{idx} th pairs ")
            if i == len(chunk_indexes)-1:
                chunks.append(args[idx:])
            else:
                chunks.append(args[idx:idx + chunk_size])
        # multiprocessing.set_start_method('spawn')
        pool.map()

    # for a, b in s3:
    #     Process(target=pair_link_prediction, args=(a, b, z1, z2)).start()
        # c = c + 1
        # z1.clear()
        # z2.clear()
        # d = (a, b)
        # z1 = set(g1.neighbors(a)).intersection(
        #     set(g1.neighbors(b)))
        # link_common_neighbors[d] = len(z1)
        # breakpoint()
        # z2 = set(g1.neighbors(a)).union(set(g1.neighbors(b)))
        # link_common_neighbors_union[d] = len(z2)
        # link_jc[d] = len(z1) / len(z2)
        # paa = (nx.adamic_adar_index(g1, [d]))
        # for pa in paa:
        #     link_aa[d] = pa[2]
        # link_pa[d] = len(set(g1.neighbors(a))) * \
        #     len(set(g1.neighbors(b)))
        # if nx.has_path(g1, source=a, target=b):
        #     link_shortest_path[d] = nx.shortest_path_length(
        #         g1, source=a, target=b)
        # else:
        #     link_shortest_path[d] = 0
        # y1 = path_count1(a, b)
        # link_f_i_f[d] = y1
        # y2a = path_count2a(a, b)
        # link_f_a_f[d] = y2a
        # y2b = path_count2b(a, b)
        # link_f_c_f[d] = y2b
        # y3a = path_count3a(a, b)
        # link_f_l1_f[d] = y3a
        # y3b = path_count3b(a, b)
        # link_f_l2_f[d] = y3b
        # y3c = path_count3c(a, b)
        # link_f_l3_f[d] = y3c
        # y4 = path_count4(a, b)
        # link_f_r_f[d] = y4
        # y6 = path_count6(a, b)
        # link_f_i_d_i_f[d] = y6
        # y7 = path_count7(a, b)
        # link_f_d_i_f[d] = y7
        # y8 = path_count8(a, b)
        # link_f_a_i_f[d] = y8
        # y9 = path_count9(a, b)
        # link_f_i_c_i_f[d] = y9
        # y11 = path_count11(a, b)
        # link_f_i_a_i_f[d] = y11
        # y12 = path_count12(a, b)
        # link_f_i_r_i_f[d] = y12
        # if g2.has_edge(a, b):
        #     link_class_variable[d] = 1
        # else:
        #     link_class_variable[d] = 0

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
    for e in link_common_neighbors:
        result_file.write(str(e))
        result_file.write(",")
        result_file.write(str(link_common_neighbors[e]))
        result_file.write(",")
        result_file.write(str(link_common_neighbors_union[e]))
        result_file.write(",")
        result_file.write(str(link_jc[e]))
        result_file.write(",")
        result_file.write(str(link_aa[e]))
        result_file.write(",")
        result_file.write(str(link_pa[e]))
        result_file.write(",")
        result_file.write(str(link_shortest_path[e]))
        result_file.write(",")
        result_file.write(str(link_f_i_f[e]))
        result_file.write(",")
        result_file.write(str(link_f_a_f[e]))
        result_file.write(",")
        result_file.write(str(link_f_c_f[e]))
        result_file.write(",")
        result_file.write(str(link_f_l1_f[e]))
        result_file.write(",")
        result_file.write(str(link_f_l2_f[e]))
        result_file.write(",")
        result_file.write(str(link_f_l3_f[e]))
        result_file.write(",")
        result_file.write(str(link_f_r_f[e]))
        result_file.write(",")
        result_file.write(str(link_f_i_d_i_f[e]))
        result_file.write(",")
        result_file.write(str(link_f_d_i_f[e]))
        result_file.write(",")
        result_file.write(str(link_f_a_i_f[e]))
        result_file.write(",")
        result_file.write(str(link_f_i_c_i_f[e]))
        result_file.write(",")
        result_file.write(str(link_f_i_a_i_f[e]))
        result_file.write(",")
        result_file.write(str(link_f_i_r_i_f[e]))
        result_file.write(",")
        result_file.write(str(link_class_variable[e]))
        result_file.write(",")
        result_file.write("\n")
    result_file.close()
    del csv_file
    del result_file
    link_common_neighbors.clear()
    link_common_neighbors_union.clear()
    link_jc.clear()
    link_aa.clear()
    link_pa.clear()
    link_shortest_path.clear()
    link_f_i_f.clear()
    link_f_a_f.clear()
    link_f_c_f.clear()
    link_f_l1_f.clear()
    link_f_l2_f.clear()
    link_f_l3_f.clear()
    link_f_r_f.clear()
    link_f_i_d_i_f.clear()
    link_f_d_i_f.clear()
    link_f_a_i_f.clear()
    link_f_i_c_i_f.clear()
    link_f_i_a_i_f.clear()
    link_f_i_r_i_f.clear()
    link_class_variable.clear()
    print("\n\n---------------------------------------------------------------")


def write_header(csv_file):
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


# def pair_link_prediction(pair):
#     project = pair[2]
#     a, b = pair[0], pair[1]
#     z1 = set()
#     z2 = set()
#     d = (a, b)
#     z1 = set(g1.neighbors(a)).intersection(set(g1.neighbors(b)))
#     link_common_neighbors[d] = len(z1)
#     breakpoint()
#     z2 = set(g1.neighbors(a)).union(set(g1.neighbors(b)))
#     link_common_neighbors_union[d] = len(z2)
#     link_jc[d] = len(z1) / len(z2)
#     paa = (nx.adamic_adar_index(g1, [d]))
#     for pa in paa:
#         link_aa[d] = pa[2]
#     link_pa[d] = len(set(g1.neighbors(a))) * \
#         len(set(g1.neighbors(b)))
#     if nx.has_path(g1, source=a, target=b):
#         link_shortest_path[d] = nx.shortest_path_length(
#             g1, source=a, target=b)
#     else:
#         link_shortest_path[d] = 0
#     y1 = path_count1(a, b)
#     link_f_i_f[d] = y1
#     y2a = path_count2a(a, b)
#     link_f_a_f[d] = y2a
#     y2b = path_count2b(a, b)
#     link_f_c_f[d] = y2b
#     y3a = path_count3a(a, b)
#     link_f_l1_f[d] = y3a
#     y3b = path_count3b(a, b)
#     link_f_l2_f[d] = y3b
#     y3c = path_count3c(a, b)
#     link_f_l3_f[d] = y3c
#     y4 = path_count4(a, b)
#     link_f_r_f[d] = y4
#     y6 = path_count6(a, b)
#     link_f_i_d_i_f[d] = y6
#     y7 = path_count7(a, b)
#     link_f_d_i_f[d] = y7
#     y8 = path_count8(a, b)
#     link_f_a_i_f[d] = y8
#     y9 = path_count9(a, b)
#     link_f_i_c_i_f[d] = y9
#     y11 = path_count11(a, b)
#     link_f_i_a_i_f[d] = y11
#     y12 = path_count12(a, b)
#     link_f_i_r_i_f[d] = y12
#     if g2.has_edge(a, b):
#         link_class_variable[d] = 1
#     else:
#         link_class_variable[d] = 0
#     # write to file
#     csv_file = f"F://{project}_no_fatty_commits_training.csv"
#     file = open(csv_file, "w")
#     with file:
#         writer = csv.writer(file)
#         write_header(file)

#     with file:
#         writer = csv.writer(file)
#         for e in link_common_neighbors:
#             row = [str(e),
#                    str(link_common_neighbors[e]),
#                    str(link_common_neighbors_union[e]),
#                    str(link_jc[e]),
#                    str(link_aa[e]),
#                    str(link_pa[e]),
#                    str(link_shortest_path[e]),
#                    str(link_f_i_f[e]),
#                    str(link_f_a_f[e]),
#                    str(link_f_c_f[e]),
#                    str(link_f_l1_f[e]),
#                    str(link_f_l2_f[e]),
#                    str(link_f_l3_f[e]),
#                    str(link_f_r_f[e]),
#                    str(link_f_i_d_i_f[e]),
#                    str(link_f_d_i_f[e]),
#                    str(link_f_a_i_f[e]),
#                    str(link_f_i_c_i_f[e]),
#                    str(link_f_i_a_i_f[e]),
#                    str(link_f_i_r_i_f[e]),
#                    str(link_class_variable[e])
#                    ]
#             writer.writerows(row)


def spawn_link_prediction_pool(project):
    # breakpoint()
    print(f"starting {project}")
    # args found
    args = [(ele[0], ele[1], project) for ele in s3]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(pair_link_prediction, args)

    # result_file.close()
    # del csv_file
    # del result_file
    link_common_neighbors.clear()
    link_common_neighbors_union.clear()
    link_jc.clear()
    link_aa.clear()
    link_pa.clear()
    link_shortest_path.clear()
    link_f_i_f.clear()
    link_f_a_f.clear()
    link_f_c_f.clear()
    link_f_l1_f.clear()
    link_f_l2_f.clear()
    link_f_l3_f.clear()
    link_f_r_f.clear()
    link_f_i_d_i_f.clear()
    link_f_d_i_f.clear()
    link_f_a_i_f.clear()
    link_f_i_c_i_f.clear()
    link_f_i_a_i_f.clear()
    link_f_i_r_i_f.clear()
    link_class_variable.clear()
    print("\n\n---------------------------------------------------------------")


if __name__ == "__main__":
    st = datetime.now()
    print(f"start time : {datetime.now()}")
    print("\nfile set of paths building")
    build_file_path()
    print("\nCommit set of files")
    build_commit_set_of_files()
    print("\nCommit time building")
    build_commit_time()
    print("\nfiles set of commits building")
    build_file_commits()
    print("\ncommit set of issues building")
    build_commit_issue()
    print("\n1.file set of issues building")
    build_file_issue()
    print("\ncommit set of authors building")
    build_commit_author()
    print("\n2a.file set of authors building")
    build_file_developer_2a()
    print("\ncommit set of committers building")
    build_commit_committer()
    print("\n2b.file set of committers building")
    build_file_developer2b()
    print("\n3a.file level 1 mapping")
    build_file_level1()
    print("\n3b.file level 2 mapping")
    build_file_level2()
    print("\n3c.file level 3 mapping")
    build_file_level3()
    print("\ncommit set of refactor building")
    build_commit_refactor()
    print("\n4.file set of refactor building")
    build_file_refactor()
    print("\nissue set of commenter's building")
    build_issue_developer()
    print("\n6.file commenter mapping building")
    file_developer_mapping()
    print("\nissue set of assignees building")
    build_issue_assignee()
    print("\nfile set of assignees building")
    build_file_assignee()
    print("\n7.file assignee mapping through issue")
    build_file_assignee_mapping()
    print("\nissue set of authors building")
    build_issue_developer8()
    print("\nfile set of authors building")
    build_file_developer8()
    print("\n8.file author mapping through issue")
    build_file_developer8_mapping()
    print("\nissue set of components building")
    build_issue_component()
    print("\n9.file component mapping building")
    file_component_mapping()
    print("\n11.file assignee 11 mapping building")
    file_developer11_mapping()
    print("\nissue set of reporters building")
    build_issue_reporter()
    print("\n12.file reporter mapping building")
    file_developer12_mapping()
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
        build_commit_set_of_src_files(name_of_project)
        print_data(name_of_project)
        build_training_file1(name_of_project)
        build_training_file2(name_of_project)
        compare_graph(name_of_project)
    print(f"Start time : {st}")
    print(f"End time : {datetime.now()}")
