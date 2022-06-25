
from pymongo import MongoClient


class Task:

    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        # Access database
        self.db = self.client["smartshark"]
        # Access collection of the database
        self.commit_project = self.db["commit_with_project_info"]
        self.commit_file = self.db["file_action"]
        self.commit_project_records = list(self.commit_project.find({}, {}))
        self.commit_file_records = list(self.commit_file.find({}))
        self.commit_set_of_files = dict()
        self.commit_project_data = dict()
        self.project_file = dict()

    def build_commit_project_data(self):
        for element in self.commit_project_records:
            commit_id = element["_id"]
            project_name = element["project_name_info"]["name"]
            if commit_id not in self.commit_project_data.keys():
                self.commit_project_data[commit_id] = project_name

    def build_commit_set_of_files(self):
        for element in self.commit_file_records:
            file_id = element["file_id"]
            commit_id = element["commit_id"]
            if commit_id not in self.commit_set_of_files.keys():
                self.commit_set_of_files[commit_id] = set()
            self.commit_set_of_files[commit_id].add(file_id)

    def build_project_file(self):
        for commit in self.commit_project_data:
            if self.commit_project_data[commit] not in self.project_file:
                self.project_file[self.commit_project_data[commit]] = dict()
            if commit in self.commit_set_of_files:
                ordered_pairs_of_files = {(x, y) for x in self.commit_set_of_files[commit] for y in
                                          self.commit_set_of_files[commit] if x != y}
                self.update_network_files(ordered_pairs_of_files, self.commit_project_data[commit])

    def update_network_files(self, ordered_pairs, project):
        for set_element in ordered_pairs:
            if set_element[0] not in self.project_file[project]:
                self.project_file[project][set_element[0]] = set()
            if set_element[1] not in self.project_file[project]:
                self.project_file[project][set_element[1]] = set()
            self.project_file[project][set_element[0]].add(set_element[1])
            self.project_file[project][set_element[1]].add(set_element[0])

    def print_data(self):
        for commit in self.commit_project_data.keys():
            print(f" Commit : {commit} | Project : {self.commit_project_data[commit]} ")
        print()
        for commit in self.commit_set_of_files.keys():
            print(f" Commit : {commit} | Files : {self.commit_set_of_files[commit]} ")
        print()

    def print_graph(self):
        for project in self.project_file:
            for file in self.project_file[project]:
                print(f" the project is {project} file is {file} and the edges : {self.project_file[project][file]}")


if __name__ == "__main__":
    t = Task()
    t.build_commit_set_of_files()
    t.build_commit_project_data()
    # t.print_data()
    print("building network \n")
    t.build_project_file()
    print("done\n\n")
    t.print_graph()
