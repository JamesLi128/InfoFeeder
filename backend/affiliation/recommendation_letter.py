import pickle
import os
from utils import RecommendationLetterSeeker

root_dir = os.listdir("../data")

paper_categories = ["cs.AI", "cs.CC", "cs.LG", "math.CO", "math.OC", "stat.ML"]

affiliation_path = '../data/author_affiliation/author_affiliation_parallel.pkl'

scholar_name = 'Mateo Díaz'
result_dict = {}
for paper_category in paper_categories:
    crnt_category_collaboration_path_ls = [path for path in root_dir if paper_category in path]
    # print(paper_category)
    # print(crnt_category_collaboration_path_ls)
    most_recent_collaboration_path = max(crnt_category_collaboration_path_ls)
    most_recent_collaboration_path = os.path.join("../data", most_recent_collaboration_path)
    letter_seeker = RecommendationLetterSeeker(author_affiliation_path=affiliation_path, collaboration_graph_path=most_recent_collaboration_path)
    _, collaborators, affiliations = letter_seeker.from_scholar_name2affiliation(scholar_name=scholar_name)
    # print(affiliations)
    # exit()
    result_dict[paper_category] = (collaborators, affiliations)

collaborator_ls, affiliation_ls = letter_seeker.combine_results(result_dict)
letter_seeker.beautiful_print(scholar_name=scholar_name, collaborators=collaborator_ls, affiliations=affiliation_ls)