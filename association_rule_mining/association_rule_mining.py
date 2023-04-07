import math
import numpy as np
import time
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from collections import defaultdict
from pymongo import MongoClient
import multiprocessing as mp


def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print(f'{f.__name__} {args} took: {te-ts} sec')
        return result

    return timed

# KAFKA, SPARK, BIGTOP, DELTA SPIKE
# # giraph
# kafka
# deltaspike
# ranger
# bigtop



class AssociationMining:
    def __init__(self, project_name, min_support=0.05, min_confidence=0.05):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["smartshark"]
        self.files = self.db["file"]
        self.file_action = self.db["file_action"]
        self.commit_with_project_info = self.db["commit_with_project_info"]
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.project_name = project_name
        self.transactions = self.get_transactions()

    def get_commit_project_info(self, project_name):
        project_commits = self.commit_with_project_info.find({"project_name_info.name" : project_name}, {"_id" : 1})
        return list(project_commits)
    
    @timeit
    def get_transactions(self):
        commit_project_list = self.get_commit_project_info(self.project_name)
        commit_project_list_1 = [d['_id'] for d in commit_project_list]
        query_file_action = list(self.file_action.find(
            {}, {"commit_id": 1, "file_id": 1}))
        #breakpoint()
        pool = mp.Pool(mp.cpu_count())
        chunks = np.array_split(query_file_action, mp.cpu_count())
        filtered_list = []
        for chunk in chunks:
            res = pool.apply_async(filter_list, args=(chunk, commit_project_list_1))
            filtered_list.extend(res.get())

        pool.close()
        pool.join()

        print(f'filtered commits count : {len(filtered_list)}')
        #filtered_file_action = [commit_file_pair for commit_file_pair in query_file_action if commit_file_pair['commit_id'] in commit_project_list]
        #print(f'filtered commits count : {len(filtered_file_action)}')
        commit_file_df = pd.DataFrame(filtered_list)
        grouped_df = commit_file_df.groupby('commit_id')['file_id'].agg(list)
        return grouped_df.values.tolist()

    def compute_support_count(self):
        transactions = self.transactions
        file_counts = defaultdict(int)

        for transaction in transactions:
            for file_id in transaction:
                file_counts[file_id] += 1

        return file_counts

    def get_freq_itemsets(self):
        te = TransactionEncoder()
        te_ary = te.fit_transform(self.transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        return apriori(df, min_support=self.min_support, use_colnames=True)

    def generate_assoc_rules(self) -> None:
        frequent_itemsets = self.get_freq_itemsets()
        rules = association_rules(
            frequent_itemsets, metric="support", min_threshold=0.01)
        # breakpoint()
        return rules

    def get_file_name(self, object_id):
        file_query = self.file.find({'file_id': object_id}, {
                                    "file_id": 1}).limit(1)
        file_id = list(file_query)
        return file_id

    def compute_transitive_rules(self):
        rules = self.generate_assoc_rules()
        antecedents = rules['antecedents'].values.tolist()
        consequents = rules['consequents'].values.tolist()
        rules_zip = list(zip(antecedents, consequents))
        transitive_pairs = []
        for rule1 in rules_zip:
            for rule2 in rules_zip:
                if rule1[1] == rule2[0]:
                    transitive_pairs.append((rule1[0], rule2[1]))
            time.sleep(1)
            print(
                f'\tnumber of transitive rules found --> {len(transitive_pairs)}', end='\r')


def main():
    list_of_projects = ["giraph",
                        "kafka",
                        "deltaspike",
                        "ranger",
                        "bigtop"]
    
    for project in list_of_projects:
        #print(project)
        for support in np.arange(0, 0.1, 0.01):
            
            print(f'project --> {project} min_support --> {support} # of assoc rules')
            assoc_mining = AssociationMining(project, min_support=support)
            
            print(len(assoc_mining.generate_assoc_rules()))
            time.sleep(2)
            assoc_mining.compute_transitive_rules()


def filter_list(chunk, commit_project_list):
    filtered_list = []
    for commit_file_pair in chunk:
        if commit_file_pair['commit_id'] in commit_project_list:
            filtered_list.append(commit_file_pair)
    return filtered_list

if __name__ == "__main__":
    main()
