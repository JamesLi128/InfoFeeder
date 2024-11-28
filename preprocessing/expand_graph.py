from utils import Collaboration_Graph_Scraper
import os
import pickle
from datetime import datetime

os.environ["PYTHONBUFFERED"] = "1"

anchor_author = "Amitabh Basu"
category_ls = ['cs.LG', 'cs.AI', 'math.CO', 'stat.ML', ]
save_path=f"../data/test_graph_expand_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
cache_path = "../data/test_graph.pkl"
print(save_path)
exit()
max_num_update = 10
scraper = Collaboration_Graph_Scraper(cache_path=cache_path, category_ls=category_ls, max_depth=1, max_num_update=max_num_update, save_path=save_path)

max_num_workers = 4
scraper.expand_collaboration_graph(max_workers=max_num_workers)