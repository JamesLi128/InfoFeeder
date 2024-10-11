from utils import Collaboration_Graph_Scraper
import os

os.environ["PYTHONBUFFERED"] = "1"

anchor_author = "Amitabh Basu"
category_ls = ['cs.LG', 'cs.AI', 'math.CO', 'stat.ML']
cache_path="../data/test_graph.pkl", 
scraper = Collaboration_Graph_Scraper(anchor_author=anchor_author, category_ls=category_ls, max_depth=1)

max_num_workers = 4
scraper.build_collaboration_graph_from_author(max_workers=max_num_workers)