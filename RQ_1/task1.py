import math
from enum import Enum
import itertools
import copy
import multiprocessing
#import networkx as nx
import pymongo
import pickle
#from plot_commits import plot_commit
import RQ_1.filtering_file as filtering_file
from rq1_utils import time_fn
import time

"""
Author : Yugandhar Desai

Research Question-1: How significant is the unseen coupling among the source code files?
"""


class Half(Enum):
    """
    first and second Halves of the dataset
    """
    FIRST = 1
    SECOND = 2


class DivideDataset:
    def __init__(self, project_name, data_limit=10000):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["smartshark"]
        self.project_name = project_name
        self.commit_with_project_info = self.db["commit_with_project_info"]
        self.file = self.db["file"]
        self.file_action = self.db["file_action"]
        self.project_commits = list(self.commit_with_project_info.find({"project_name_info.name": project_name},
                                                                  {"_id": 1, "committer_date": 1, "committer_id": 1}).limit(data_limit))
        self.project_commit_issue = list(self.commit_with_project_info.find({"project_name_info.name": project_name}, 
                                                                            {"_id": 1, "linked_issue_ids": 1}).limit(data_limit))
    
        self.file_commit_1 = self.make_file_commit_dict(Half.FIRST)
        self.file_commit_2 = self.make_file_commit_dict(Half.SECOND)
        self.refactoring = self.db["refactoring"]
        
        # self.commit_project = self.db["commit_with_project_info"]


        # self.commit_project_records = list(self.commit_project.find({}, {}))
        # self.commit_file = self.db["file_action"]
        # self.commit_refactoring = self.db["refactoring"]
        # self.issue_author = self.db["issue_comment"]
        # self.issue_reporter1 = self.db["issue"]
        # self.file_path = self.db["file"]
        # self.commit_issue_records = list(
        #     self.commit_project.find({}, {"_id": 1, "linked_issue_ids": 1}))
        # self.commit_file_records = list(
        #     self.commit_file.find({}, {"file_id": 1, "commit_id": 1}))
        # self.commit_refactor_records = list(
        #     self.commit_refactoring.find({}, {"_id": 1, "commit_id": 1}))
        # self.issue_author_records = list(
        #     self.issue_author.find({}, {"issue_id": 1, "author_id": 1}))
        # self.issue_reporter_records = list(
        #     self.issue_reporter1.find({}, {"_id": 1, "reporter_id": 1}))
        # self.issue_assignee_records = list(
        #     self.issue_reporter1.find({}, {"_id": 1, "assignee_id": 1}))
        # self.issue_component_records = list(
        #     self.issue_reporter1.find({}, {"_id": 1, "components": 1}))

    def get_start_date(self):
        """
        returns the start date of the project
        """
        
        sorted_project_commits = sorted(self.project_commits, key=lambda k: k['committer_date'])
        return sorted_project_commits[0]["committer_date"]

    def get_end_date(self):
        """
        returns the end date of the project
        """
        
        sorted_project_commits = sorted(self.project_commits, key=lambda k: k['committer_date'], reverse=True)        
        return sorted_project_commits[0]["committer_date"]

    def find_midpoint_date(self, start_date, end_date):
        """
        returns the midpoint date of the project
        """
        commit_date_list = []
        
        for commit in self.project_commits:
            commit_date_list.append(
                (commit['committer_id'], commit['committer_date']))
            
        commit_date_list.sort(key=lambda tup: tup[1])

        time_period = end_date - start_date
        midpoint_date = start_date + time_period / 2

        return midpoint_date

    def check_commit_half(self, commit_date, midpoint_date):
        """
        returns the half of the project
        """
        if commit_date <= midpoint_date:
            return Half.FIRST
        else:
            return Half.SECOND

    def check_commit_distribution(self):
        """
        returns the distribution of the commit in the first half and second half
        """
        first_half = 0
        second_half = 0

        midpt_date = self.find_midpoint_date(
            self.get_start_date(), self.get_end_date())

        
        for commit in self.project_commits:
            if self.check_commit_half(commit['committer_date'], midpt_date) == Half.FIRST:
                first_half += 1
            else:
                second_half += 1
        return first_half, second_half

    def get_project_files(self, half):
        """
        returns the list of dictionary of commit_id, file_id, file_path in the project
        check if the commit is in the first half or second half
        """
        commit_id_list = []
        midpt_date = self.find_midpoint_date(
            self.get_start_date(), self.get_end_date())

        

        for commit in self.project_commits:
            if self.check_commit_half(commit['committer_date'], midpt_date) == half:
                commit_id_list.append((commit['_id']))
            else:
                pass

        # commits_id_list = []
        # project_commits_copy = copy.deepcopy(self.project_commits)
        # for commit in project_commits_copy:
        #     commits_id_list.append(commit['_id'])

        # print(len(commit_id_list))

        # find files for the commit_id_list
        project_files = self.file_action.find(
            {"commit_id": {"$in": commit_id_list}}, {"_id": 1, "file_id": 1, })
        
        # filtering files 
        filtered_files = []
        for file in project_files:
            # find out file path using file collection in db
            file_path = self.file.find_one({"_id": file["file_id"]}, {"path": 1})
            if filtering_file.is_source_code_file(file_path["path"]):
                file['file_path'] = file_path["path"]
                filtered_files.append(file["file_id"])

        return filtered_files

    def make_file_commit_dict(self, half):
        """
        returns the dictionary of file_id and list of commit_id
        """
        
        file_commit_dict = {}
        commits = [commit["_id"] for commit in self.project_commits]
        file_action_list = list(self.file_action.find({"commit_id": {"$in": commits}}))
        #print(f'pre filtered file_action_list length: {len(file_action_list)}')
        for file in file_action_list:
            file_path = self.file.find_one({"_id": file["file_id"]}, {"path": 1})
            is_source_code = filtering_file.is_source_code_file(file_path["path"])
            
            midpt_date = self.find_midpoint_date(self.get_end_date(), self.get_start_date())
            # find the commit date of the commit_id in list project_commits
            commit_date = None
            for commit in self.project_commits:
                if commit["_id"] == file["commit_id"]:
                    commit_date = commit["committer_date"]
            
            if self.check_commit_half(commit_date, midpt_date) == half and is_source_code:
                file_commit_dict.setdefault(file["file_id"], []).append(file["commit_id"])
        #print(f'unique file_ids in file_commit_dict: {len(set(file_commit_dict.keys()))}')
        return file_commit_dict
    
    def make_file_issue_dict(self, half):
        """
        returns the dictionary of file_id and list of issue_id
        """
        commits = self.make_file_commit_dict(half)
        file_issue_dict = {}
        issue_set = set()

        for file_id, commit_list in commits.items():
            issue_list = []
            for commit_id in commit_list:
                # find the issue_id of the commit_id in self.project_commits_issue
                for commit in self.project_commit_issue:
                    if commit_id == commit["_id"]:
                        if "linked_issue_ids" in commit:
                            issue_list.extend(commit["linked_issue_ids"])
                            issue_set.update(commit["linked_issue_ids"])

                        #avg_issue_per_file += len(commit["linked_issue_ids"])
            file_issue_dict[file_id] = issue_list
            
        print(f'unique issue_ids in file_issue_dict for {half} : {len(issue_set)}')
        print(f'avg issue per file for {half}: {len(issue_set)/len(file_issue_dict)}')

        return file_issue_dict


    def make_all_file_pairs(self, half):
        """print(len(file_combinations))
        Make all possible pairs among them
        Select the pairs which are connected (at least one commit exists where 
        both files are committed together ) in the first half and calculate the 
        percentage of these which are also connected in the latter half
        """
        file_commit_dict = self.make_file_commit_dict(half)
        file_pairs = []
        for file_id in file_commit_dict:
            for file_id2 in file_commit_dict:
                if file_id != file_id2:
                    file_pairs.append((file_id, file_id2))

        return file_pairs

    def make_connected_file_pairs(self, co_change, half):
        """
        returns the list of connected file pairs
        """
        
        connected_file_pairs = []
        # common_file_ids = []
        # common_file_dict = dict()
        file_dict = dict()
        if co_change == 'commit':
            file_dict = self.make_file_commit_dict(half)
        elif co_change == 'issue':
            file_dict = self.make_file_issue_dict(half)

        first_half_files = set(self.get_project_files(Half.FIRST))
        second_half_files = set(self.get_project_files(Half.SECOND))
        common_files = first_half_files.intersection(second_half_files)

        files = list(file_dict.keys())

        for file_id in files:
            if file_id not in common_files:
                del file_dict[file_id]

        file_combinations = list(itertools.combinations(file_dict.keys(), 2))
        #print("combinations",len(file_combinations))
        # 4M
        #process_file_pairs(file_combinations, file_commit_dict)
        # for file_id, file_id2 in file_combinations:    
        #     if set(file_commit_dict[file_id]).intersection(file_commit_dict[file_id2]):
        #         connected_file_pairs.append((file_id, file_id2))

        chunks = []
        for i in range(0, len(file_combinations), 10000):
            chunks.append(file_combinations[i:i+10000])

        pool = multiprocessing.Pool(processes=4)
        for chunk in chunks:
            res = pool.apply_async(process_file_pairs, args=(chunk, file_dict))
            #print(f'processed {len(chunk)} file pairs\n')
            connected_file_pairs.extend(res.get())
        pool.close()
        pool.join()
        print("connected file pairs",len(connected_file_pairs))
        return connected_file_pairs
        
    def file_pair_frequency(self, half):

        file_dict = self.make_file_issue_dict(half)
        file_combinations = list(itertools.combinations(file_dict.keys(), 2))
        file_pair_frequency = dict()

        chunks = []
        for i in range(0, len(file_combinations), 10000):
            chunks.append(file_combinations[i:i+10000])

        pool = multiprocessing.Pool(processes=4)
        for chunk in chunks:
            res = pool.apply_async(process_pair_freq, args=(chunk, file_dict))
            #print(f'processed {len(chunk)} file pairs\n')
            file_pair_frequency.update(res.get())

        pool.close()
        pool.join()

        # simple statistics on file pair frequency
        print(f'number of file pairs for {half}: {len(file_pair_frequency)}')
        print(f'average frequency of file pairs for {half}: {sum(file_pair_frequency.values())/len(file_pair_frequency)}')
        print(f'max frequency of file pairs for {half}: {max(file_pair_frequency.values())}')
        print(f'min frequency of file pairs for {half}: {min(file_pair_frequency.values())}')



    def make_commit_file_dict(self, half):
        """
        returns the dictionary of commit_id and list of file_id
        """
        commit_file_dict = {}
        
        for commit in self.project_commits:
            midpt_date = self.find_midpoint_date(
                self.get_start_date(), self.get_end_date())
            
            if self.check_commit_half(commit['committer_date'], midpt_date) == half:
                #breakpoint()
                for file in self.file_action.find({"commit_id": commit['_id']}):
                    if commit['_id'] not in commit_file_dict:
                        commit_file_dict[commit['_id']] = [file['file_id']]
                    else:
                        commit_file_dict[commit['_id']].append(file['file_id'])
            else:
                pass
       
        return commit_file_dict

    def connected_file_pairs(self, half):
        """
        returns the graph of file pairs
        """
        file_pair_graph = []
        commit_file_dict = self.make_commit_file_dict(half)
        for commit in commit_file_dict:

            file_combinations = itertools.combinations(
                commit_file_dict[commit], 2)
            print(f'{len(commit_file_dict[commit])}', end='')
            print()
            for file_pair in file_combinations:
                file_pair_graph.append(file_pair[0], file_pair[1])

        return file_pair_graph

    def check_connected(self, file_pair, half):
        """
        returns True if the file pair is connected in the given half
        """

        if half == Half.FIRST:
            file_commit_dict = self.file_commit_1
        else:
            file_commit_dict = self.file_commit_2

        try:
            file1_commit_list = file_commit_dict[file_pair[0]]
            file2_commit_list = file_commit_dict[file_pair[1]]
        except KeyError:
            return False
        
        for commit in file1_commit_list:
            if commit in file2_commit_list:
                return True
        return False
    
    @time_fn
    def get_filepair_sets(self, first_half_file_pairs, second_half_file_pairs):
        first_set = set(first_half_file_pairs)
        second_set = set(second_half_file_pairs)
        connected_file_pairs = []
        # file pairs which are connected in the first half and also connected in the second half
        connected_file_pairs = first_set.intersection(second_set)
        # file pairs which are connected in the first half but not connected in the second half
        disconnected_file_pairs = first_set.difference(second_set)
        # file pairs which are not connected in the first half but connected in the second half
        newly_connected_file_pairs = second_set.difference(first_set)
        s3 = []
        s4 = []
        for file_pair in first_set:
            file_path0 = self.file.find_one({"_id": file_pair[0]}, {"path": 1})
            file_path1 = self.file.find_one({"_id": file_pair[1]}, {"path": 1})
            file_paths = (file_path0['path'], file_path1['path'])
            if is_inter_module(file_paths):
                s3.append(file_pair)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

        for file_pair in second_set:
            file_path0 = self.file.find_one({"_id": file_pair[0]}, {"path": 1})
            file_path1 = self.file.find_one({"_id": file_pair[1]}, {"path": 1})
            file_paths = (file_path0['path'], file_path1['path'])
            if is_inter_module(file_paths):
                s4.append(file_pair)

        set_dict = { 's1' : connected_file_pairs,
                     's2' : newly_connected_file_pairs,
                     's3' : s3,
                     's4' : s4}
        
        return set_dict

    @time_fn
    def file_pair_evolution(self, first_half_file_pairs, second_half_file_pairs):
        """
        Select the pairs which are connected (at least one commit exists where both files are committed together ) in the 
        first half and calculate the percentage of these which are also connected in the latter half( again, at least one 
        commit exists where both files are committed together   ). This set of pairs is designated as S1
        """
        first_set = set(first_half_file_pairs)
        second_set = set(second_half_file_pairs)
        connected_file_pairs = []
        # file pairs which are connected in the first half and also connected in the second half
        connected_file_pairs = first_set.intersection(second_set)
        # file pairs which are connected in the first half but not connected in the second half
        disconnected_file_pairs = first_set.difference(second_set)
        # file pairs which are not connected in the first half but connected in the second half
        newly_connected_file_pairs = second_set.difference(first_set)
        
        # inter module file pairs
        intermodule_file_pairs_first = []
        intermodule_file_pairs_second = []
        for file_pair in first_set:
            if is_inter_module(file_pair):
                intermodule_file_pairs_first.append(file_pair)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

        for file_pair in second_set:
            if is_inter_module(file_pair):
                intermodule_file_pairs_second.append(file_pair)

        # refactoring_pairs_1 = self.refactoring_pairs(connected_file_pairs)
        # refactoring_pairs_2 = self.refactoring_pairs(newly_connected_file_pairs)


        print(f"\t S1 \n connected file pairs in first and second -> {len(connected_file_pairs)}")
        # percentage of connected file pairs in the first half which are also connected in the second half
        print(f'\t{(len(connected_file_pairs)/len(first_set))*100} %')
        print('-'*60)
        print(f"\tconnected file pairs in first, but not second -> {len(disconnected_file_pairs)}")
        print('-'*60)
        print(f"\t S2 \n connected file pairs in second, but not first -> {len(newly_connected_file_pairs)}")
        print('-'*60)
        print(f"\t S3 \n intermodule file pairs in first -> {(len(intermodule_file_pairs_first)/len(first_set))*100} %")
        print('-'*60)
        print(f"\tS4 \n intermodule file pairs in second -> {(len(intermodule_file_pairs_second)/len(second_set))*100} %")
        print('-'*60)

    def check_refactoring_commit(self, commit_id):
        if self.refactoring.find_one({"commit_id": commit_id}):
            return True
        return False


    def refactoring_pairs(self, file_pair_set, half):

        file_commit_dict = {}
        if half == Half.FIRST:
            file_commit_dict = self.file_commit_1
        else:
            file_commit_dict = self.file_commit_2

        # traverse the file pair set, get the commit list for each file pair
        # and check if any of the commit is a refactoring commit
        refactoring_pair = []
        for file_pair in file_pair_set:
            file1_commit_list = file_commit_dict[file_pair[0]]
            file2_commit_list = file_commit_dict[file_pair[1]]
            for commit in file1_commit_list:
                if commit in file2_commit_list:
                    if self.check_refactoring_commit(commit):
                        refactoring_pair.append(file_pair)
                        break

        return refactoring_pair
                    

def process_file_pairs(chunk, file_commit_dict):
    """
    helper function for make_connected_file_pairs method
    """
    connected_file_pairs = []
    for file_id, file_id2 in chunk:
        if set(file_commit_dict[file_id]).intersection(file_commit_dict[file_id2]):
            connected_file_pairs.append((file_id, file_id2))

    return connected_file_pairs

def process_pair_freq(chunk, file_dict):

    freq_dict = {}
    for file_id, file_id2 in chunk:
        freq_dict[(file_id, file_id2)] = set(file_dict[file_id]).intersection(file_dict[file_id2])

    return freq_dict

def is_inter_module(file_pair):
    """
    returns True if the file pair is inter module
    """
    
    file1_module = file_pair[0].split('/')[-2]
    file2_module = file_pair[1].split('/')[-2]
    if file1_module != file2_module:
        return True 
    return False


if __name__ == "__main__":
    projects = ['mahout', 'pdfbox', 'opennlp', 'openwebbeans', 'mina-sshd', 'helix', 'curator',
     'storm', 'cxf-fediz',
     'knox', 'zeppelin', 'samza',
     'directory-kerby', 'pig', 'manifoldcf',
     'giraph', 'bigtop', 'kafka', 'oozie',
     'falcon', 'deltaspike', 'calcite',
     'parquet-mr', 'tez', 'lens', 'phoenix',
     'kylin', 'ranger']
    
    for idx, project in enumerate(projects):

        start = time.time()
        print(f"--- {idx}. {project} ---")
        divide_dataset = DivideDataset(project)
        start_date = divide_dataset.get_start_date()
        end_date = divide_dataset.get_end_date()
        # format fstrings for date
        # print(f"start date  ->  {start_date}")
        # print(f"end date  ->  {end_date}")
        # print(f"mid point date -> {divide_dataset.find_midpoint_date(start_date, end_date)}")
        # print(f" commit distribution (first half, second half) -> {divide_dataset.check_commit_distribution()}")
        # print(len(divide_dataset.get_project_files(Half.SECOND)))
        # print(f' length of commit-file dict -> {len(divide_dataset.make_commit_file_dict(Half.SECOND))}')
        # print(f" length of file-commit dict -> {len(divide_dataset.make_file_commit_dict(Half.SECOND))}")
        # first_half_file_pairs = divide_dataset.make_connected_file_pairs('issue', Half.FIRST)
        # second_half_file_pairs = divide_dataset.make_connected_file_pairs('issue', Half.SECOND)
        # filter the file pairs with files which are not common in both halves

        first_half_file_pairs = divide_dataset.make_connected_file_pairs('issue', Half.FIRST)
        second_half_file_pairs = divide_dataset.make_connected_file_pairs('issue', Half.SECOND)

        # first_half_files = set(divide_dataset.get_project_files(Half.FIRST))
        # second_half_files = set(divide_dataset.get_project_files(Half.SECOND))
        # common_files = first_half_files.intersection(second_half_files)
        # file_pair_1 = []
        # file_pair_2 = []
        # for file_pair in first_half_file_pairs:
        #     if file_pair[0] in common_files and file_pair[1] in common_files:
        #         file_pair_1.append(file_pair)
        # for file_pair in second_half_file_pairs:
        #     if file_pair[0] in common_files and file_pair[1] in common_files:
        #         file_pair_2.append(file_pair)

        set_dict = divide_dataset.get_filepair_sets(first_half_file_pairs, second_half_file_pairs)
        # s1 = set_dict.get('s1')
        # s2 = set_dict.get('s2')
        # s3 = set_dict.get('s3')
        # s4 = set_dict.get('s4')

        with open(f'..RQ_2/proj_file_pair_sets/{project}_file_pair_sets.pickle', 'wb') as f:
            # pickle sets_dict to file
            pickle_msg = f'pickling {project} file pair sets'
            print(pickle_msg)
            pickle.dump(set_dict, f)

        end = time.time()
        time_taken = end - start
        print(f'time taken for {project} -> {time_taken}')
        
        #divide_dataset.file_pair_evolution(file_pair_1, file_pair_2)
        #print(divide_dataset.make_connected_file_pairs(Half.SECOND)[:10])
        # divide_dataset.file_pair_evolution()

        # print(len(divide_dataset.connected_file_pairs(Half.SECOND)))
        # plot_commit('giraph')
        # print(len(divide_dataset.connected_file_pairs(Half.SECOND)))

        # mongoshell command to get files which are in list of commitsissue
        # db.
