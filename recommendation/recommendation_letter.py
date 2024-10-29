import pickle
import os
from utils import RecommendationLetterSeeker

affiliation_path = '../data/author_affiliation/author_affiliation_parallel.pkl'
collaboration_graph_path = '../data/test_graph_expand.pkl'
scholar_name = 'Amitabh Basu'
letter_seeker = RecommendationLetterSeeker(author_affiliation_path=affiliation_path, collaboration_graph_path=collaboration_graph_path)
_, collaborators, affiliations = letter_seeker.from_scholar_name2affiliation(scholar_name=scholar_name)
letter_seeker.beautiful_print(scholar_name=scholar_name, collaborators=collaborators, affiliations=affiliations)