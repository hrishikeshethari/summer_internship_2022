from pymongo import MongoClient
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import numpy as np

# giraph
# kafka
# deltaspike
# ranger
# bigtop


def plot_commit(project_name):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["smartshark"]
    commit_with_project_info = db["commit_with_project_info"]
    project_commits = commit_with_project_info.find(
        {"project_name_info.name": project_name}, {"_id": 1, "committer_date": 1}).sort('committer_date', 1)
    project_commits_list = list(project_commits)

    monthly_commits = defaultdict(int)
    for commit in project_commits_list:
        commit_date = commit["committer_date"]
        month_key = commit_date.strftime("%b %y")
        monthly_commits[month_key] += 1

    x_values = list(monthly_commits.keys())
    y_values = list(monthly_commits.values())

    plt.figure(figsize=(50, 8))
    plt.bar(x_values, y_values)
    plt.xlabel("Date")
    plt.ylabel("Number of Commits")

    num_ticks = len(x_values)
    # increase spacing by 3
    tick_positions = [i for i in range(num_ticks) if i % 2 == 0]
    tick_labels = [x_values[i] for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)

    plt.title(f"{project_name} Monthly Commits")
    plt.show()
    time.sleep(2)
    plt.savefig()


def main():

    # PROJECTS
    list_of_projects = ["giraph",
                        "kafka",
                        "deltaspike",
                        "ranger",
                        "bigtop"]

    for project in list_of_projects:
        plot_commit(project)


if __name__ == "__main__":
    main()
