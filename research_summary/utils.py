import os
import pickle
import torch
import re
from preprocessing.utils import Collaboration_Graph_Scraper
from tqdm import tqdm

class Research_Summary_Generator:
    def __init__(self, data_root_dir : str):
        self.data_root_dir = data_root_dir


    def _collaboration_graph_file_name_matcher(self, data_root_dir : str):
        file_name_ls = os.listdir(data_root_dir)
        regex_pattern = r"^([a-zA-Z]+\.[a-zA-Z]+)_collaboration_graph_(\d{8}_\d{6})\.pkl$"
        category_datetime_ls = []
        for file_name in file_name_ls:
            match = re.match(regex_pattern, file_name)
            if match:
                category_datetime_ls.append((match.group(1), match.group(2)))
