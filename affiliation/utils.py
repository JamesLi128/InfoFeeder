import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import pickle
import networkx as nx

from tqdm import tqdm

class Paper:
    def __init__(self, paper_id : str, scholar_name_set : set[str], root_dir : str = "../data/"):
        self.paper_id = paper_id
        self.scholar_name_set = scholar_name_set
        self.root_dir = root_dir
        self.paper_name = paper_id2paper_name(paper_id)
        self.embeddings = self._load_embeddings()
    
    def _load_embeddings(self):
        embeddings_path = os.path.join(self.root_dir, "embeddings", self.paper_name + ".pt")
        embeddings = torch.load(embeddings_path)
        return embeddings
    
class Scholar:
    def __init__(self, scholar_name : str, Paper_set : set[Paper]):
        self.scholar_name = scholar_name
        self.Paper_set = Paper_set

    def add_paper(self, paper : Paper):
        self.Paper_set.add(paper)

    def add_many_papers(self, papers : set[Paper]):
        papers = set(papers)
        self.Paper_set.update(papers)

    def get_paper_set(self):
        return self.Paper_set
    
    def num_papers(self):
        return len(self.Paper_set)

class User:
    def __init__(self, Scholar_set : set[Scholar], Paper_set : set[Paper]):
        self.Scholar_set = Scholar_set
        self.Paper_set = Paper_set

    def add_scholar(self, scholar : Scholar):
        self.Scholar_set.add(scholar)

    def add_many_scholars(self, scholars : set[Scholar]):
        scholars = set(scholars)
        self.Scholar_set.update(scholars)

    def add_paper_by_id(self, paper_id : str, scholar_name_set : set[str]):
        paper = Paper(paper_id, scholar_name_set)
        self.Paper_set.add(paper)

    def add_paper(self, paper : Paper):
        self.Paper_set.add(paper)

    def add_many_papers(self, papers : set[Paper]):
        papers = set(papers)
        self.Paper_set.update(papers)

    def get_scholar_set(self):
        return self.Scholar_set
    
    def num_papers(self):
        return len(self.Paper_set)
    
    def num_scholars(self):
        return len(self.Scholar_set)
    
    def get_paper_set(self):
        return self.Paper_set

class RecommendationLetterSeeker: 
    '''
    Based on affiliation dictionary, collaboration graph, and given info, recommend scholars to write recommendation letters.
    '''
    def __init__(self, author_affiliation_path : str, collaboration_graph_path : str):
        _, self.affiliation_dict = self._load_affiliation_dict(author_affiliation_path)
        self.collaboration_graph, self.paper_id2author_ls = self._load_collaboration_graph(collaboration_graph_path)

    def _load_affiliation_dict(self, path : str):
        with open(path, 'rb') as f:
            affiliation_dict = pickle.load(f)
        return affiliation_dict
    
    def _load_collaboration_graph(self, path : str):
        with open(path, 'rb') as f:
            collaboration_graph, paper_id2author_ls, _, _ = pickle.load(f)
        return collaboration_graph, paper_id2author_ls

    def from_scholar_name2affiliation(self, scholar_name : str):
        if scholar_name not in self.collaboration_graph.nodes:
            print(f"Query: scholar '{scholar_name}' not found in collaboration graph.")
            return None

        collaborators = list(self.collaboration_graph[scholar_name].keys())
        affiliations = []
        for collaborator in collaborators:
            collaborator = collaborator.lower()
            if collaborator in self.affiliation_dict:
                rst = self.affiliation_dict[collaborator]
                institution_ls = list(rst.keys())
                departments_ls = list(rst.values())
            else:
                institution_ls = []
                departments_ls = []
            affiliations.append((institution_ls, departments_ls))
        return (scholar_name, collaborators, affiliations)
    
    def beautiful_print(self, scholar_name : str, collaborators : list[str], affiliations : list[tuple[str, list[str]]]):
        print(f"Scholar: {scholar_name}")
        print("Collaborators and their affiliations:")
        for collaborator, (institution_ls, departments_ls) in zip(collaborators, affiliations):
            institution_ls = [i for i in institution_ls if i != "N/A"]  # Filter out empty institutions
            print(f" - {collaborator}: ")
            if len(institution_ls) == 0 and len(departments_ls) == 0:
                print(f"\tNo affiliation found.")
                continue
            for institution, departments in zip(institution_ls, departments_ls):
                departments = [d for d in departments if d != "N/A"]  # Filter out empty departments
                print(f"\t{institution}, \tDepartments: {', '.join(departments)}")


def paper_id2paper_name(paper_id : str) -> str:
    return paper_id.replace('/', '_')

def paper_name2paper_id(paper_name : str) -> str:
    return paper_name.replace('_', '/')