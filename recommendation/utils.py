import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os

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


def paper_id2paper_name(paper_id : str) -> str:
    return paper_id.replace('/', '_')