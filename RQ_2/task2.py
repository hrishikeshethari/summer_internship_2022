import pymongo
import multiprocessing as mp
from RQ_1.task1 import DivideDataset, Half


class Task2:
    def __init__(self, dividedataset: DivideDataset, **kwargs):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client['smartshark']
        self.refactoring = self.db['refactoring']
        self.file_action = self.db['file_action']
        self.commit_with_project_info = self.db['comit_with_project_info']
        self.ref_list = list(self.refactoring.find())
        self.s1 = kwargs.get('s1')
        self.s2 = kwargs.get('s2')
        self.s3 = kwargs.get('s3')
        self.s4 = kwargs.get('s4')
        self.file_commit_1 = dividedataset.make_file_commit_dict(Half.FIRST)
        self.file_commit_2 = dividedataset.make_file_commit_dict(Half.SECOND)
        self.file_commit_dict = {**self.file_commit_1, **self.file_commit_2}
        self.ref_list = list(self.refactoring.find())
        self.ref_dict = { ref["commit_id"] : (ref["type"], ref["description"]) for ref in self.ref_list }

        self.high_impact_types = {'extract_superclass', 'extract_and_move_method', 'extract_class'
                                                                                   'move_method', 'pull_up_method',
                                  'pull_down_method',
                                  'extract_subclass', 'extract_interface', 'move_and_rename_class',
                                  'move_and_rename_attribute', 'move_and_rename_class', 'move_attribute',
                                  'move_class', 'move_method', 'pull_up_method', 'pull_down_method',
                                  'pull_down_attribute',
                                  'pull_up_attribute', 'rename_class'}

        self.low_impact_types = self.get_low_impact_types()

    def get_low_impact_types(self) -> set:
        low_impact_types = set()

        for refactoring in self.ref_list:
            if refactoring['type'] not in self.high_impact_types:
                low_impact_types.add(refactoring['type'])

        return low_impact_types
<<<<<<< HEAD
    
    def check_refactoring_commit(self, commit_id : str) -> bool:
        q = self.refactoring.find_one({'commit_id' : commit_id})
        if q is not None:
            return True
        return False
            
    def refactoring_commits(self, pairs, common_commits : bool, print_mode : bool) -> None:
=======

    def check_refactoring_commit(self, commit_id: str) -> bool:
        # ObjectId("5b23e067a1501a2542d12fc8") ObjectId("5b23e067a1501a2542d12fc8"),
        # q = self.refactoring.find_one({"commit_id": commit_id})
        if commit_id in self.ref_dict:
            return True
        return False

    def refactoring_commits(self, pairs, common_commits: bool, print_mode: bool):
>>>>>>> 6b3191bc746e6e3cd869ae144c784a083e221921

        refactoring_commits = []

        for file1, file2 in pairs:
            commit_list1 = self.file_commit_dict.get(file1)
            commit_list2 = self.file_commit_dict.get(file2)
            if commit_list1 is None and commit_list2 is None:
                continue
            if common_commits:
                common_commits = set(commit_list1).intersection(set(commit_list2))
                for commit in common_commits:
                    if self.check_refactoring_commit(commit):
                        refactoring_commits.append((file1, file2, commit))
                        break
            else:
                for commit1 in commit_list1:
                    # breakpoint()
                    if self.check_refactoring_commit(commit1):
                        refactoring_commits.append((file1, file2, commit1))

                    for commit2 in commit_list2:
                        if self.check_refactoring_commit(commit2):
                            refactoring_commits.append((file1, file2, commit2))

        if not print_mode:
            return refactoring_commits
        else:
            if common_commits:
                print(f'---- Common commits between file pairs ----')
            else:
                print(f'---- All commits for file pairs ----')

        unique_pairs = set([(file1, file2) for file1, file2, _commit in refactoring_commits])
        percentage = len(unique_pairs) / len(pairs)
        print(f"Percentage of refactoring commits: {percentage}")

    def refactoring_commits_distribution(self, commits: list) -> None:

        ref_types = self.high_impact_types.union(self.low_impact_types)
        #ref_types_list = list(ref_types.keys())
        distribution_dict = {refactoring: 0 for refactoring in ref_types}

        for commit in commits:
            #q = self.refactoring.find_one({'commit_id': commit}, {'type': 1})
            comm = self.ref_dict[commit]
            ref_type = comm[0]
            ref_type = ref_type.lower()
            if ref_type in ref_types:
                distribution_dict[ref_type] += 1
            else:
                distribution_dict[ref_type] += 1

        print(f'Distribution of refactoring commits: ')
        print('-' * 50)
        print(f'High impact refactoring types: ')
        for high_impact_type in self.high_impact_types:
            print(f'{high_impact_type} : {distribution_dict[high_impact_type]}')
        print('-' * 50)
        print(f'Low impact refactoring types: ')
        for low_impact_type in self.low_impact_types:
            print(f'{low_impact_type} : {distribution_dict[low_impact_type]}')
        print('-' * 50)


def levenshtein_distance(str1, str2):
    distances = [[0 for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        distances[i][0] = i
    for j in range(1, len(str2) + 1):
        distances[0][j] = j

    for j in range(1, len(str2) + 1):
        for i in range(1, len(str1) + 1):
            if str1[i - 1] == str2[j - 1]:
                distances[i][j] = distances[i - 1][j - 1]
            else:
                distances[i][j] = 1 + min(distances[i - 1][j],
                                          distances[i][j - 1],
                                          distances[i - 1][j - 1])

    return distances[-1][-1]


def similar(str1, str2, max_distance=3):
    distance = levenshtein_distance(str1, str2)
    return distance <= max_distance


if __name__ == '__main__':

    project = ['mahout', 'pdfbox', 'opennlp', 'openwebbeans', 'mina-sshd', 'helix', 'curator',
     'storm', 'cxf-fediz',
     'knox', 'zeppelin', 'samza',
     'directory-kerby', 'pig', 'manifoldcf',
     'giraph', 'bigtop', 'kafka', 'oozie',
     'falcon', 'deltaspike', 'calcite',
     'parquet-mr', 'tez', 'lens', 'phoenix',
     'kylin', 'ranger']

<<<<<<< HEAD
    task2 = Task2(divide_dataset, s1=s1, s2=s2, s3=s3, s4=s4)
    task2.refactoring_commits(s1, common_commits=False, print_mode=True)
=======
    for proj in project:
        print(f"--- {proj} ---")
        divide_dataset = DivideDataset(proj)
        first_half_file_pairs = divide_dataset.make_connected_file_pairs('issue', Half.FIRST)
        second_half_file_pairs = divide_dataset.make_connected_file_pairs('issue', Half.SECOND)
        set_dict = divide_dataset.get_filepair_sets(first_half_file_pairs, second_half_file_pairs)
        s1 = set_dict.get('s1')
        s2 = set_dict.get('s2')
        s3 = set_dict.get('s3')
        s4 = set_dict.get('s4')
>>>>>>> 6b3191bc746e6e3cd869ae144c784a083e221921

        task2 = Task2(divide_dataset, s1=s1, s2=s2, s3=s3, s4=s4)
        task2.refactoring_commits(s1, common_commits=True, print_mode=True)

        # commits for s1
        print('--- S1 ---')
        file_commits = task2.refactoring_commits(s1, common_commits=False, print_mode=False)
        commits_only = [commit[2] for commit in file_commits]

<<<<<<< HEAD
    # s2
    task2.refactoring_commits(s2, common_commits=False, print_mode=True)

    file_commits = task2.refactoring_commits(s2, common_commits=False, print_mode=False)
    commits_only_2 = [commit[2] for commit in file_commits]

    task2.refactoring_commits_distribution(commits_only_2)

    # s3
    task2.refactoring_commits(s3, common_commits=False, print_mode=True)

    file_commits = task2.refactoring_commits(s3, common_commits=False, print_mode=False)
    commits_only_3 = [commit[2] for commit in file_commits]

    task2.refactoring_commits_distribution(commits_only_3)

    # s4
    task2.refactoring_commits(s4, common_commits=False, print_mode=True)

    file_commits = task2.refactoring_commits(s4, common_commits=False, print_mode=False)
    commits_only_4 = [commit[2] for commit in file_commits]

    task2.refactoring_commits_distribution(commits_only_4)



=======
        task2.refactoring_commits_distribution(commits_only)
        # s2
        print('--- S2 ---')
        task2.refactoring_commits(s2, common_commits=False, print_mode=True)
>>>>>>> 6b3191bc746e6e3cd869ae144c784a083e221921

        file_commits = task2.refactoring_commits(s2, common_commits=False, print_mode=False)
        commits_only_2 = [commit[2] for commit in file_commits]

        task2.refactoring_commits_distribution(commits_only_2)

        # s3
        print('--- S3 ---')
        task2.refactoring_commits(s3, common_commits=False, print_mode=True)

        file_commits = task2.refactoring_commits(s3, common_commits=False, print_mode=False)
        commits_only_3 = [commit[2] for commit in file_commits]

        task2.refactoring_commits_distribution(commits_only_3)

        # s4
        print('--- S4 ---')
        task2.refactoring_commits(s4, common_commits=False, print_mode=True)

        file_commits = task2.refactoring_commits(s4, common_commits=False, print_mode=False)
        commits_only_4 = [commit[2] for commit in file_commits]

        task2.refactoring_commits_distribution(commits_only_4)

