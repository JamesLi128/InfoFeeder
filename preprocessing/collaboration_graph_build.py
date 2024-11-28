from utils import url_generator, Collaboration_Graph_Scraper, visualize_collaboration_graph_matplotlib
from datetime import datetime
import os

category_ls = ['math.ST']
scholar_name = "Mateo Díaz"


for category in category_ls:
    print(f"Expanding {scholar_name}'s collaboration graph in: {category_ls}")
    save_path = f"../data/{'_'.join([category])}_collaboration_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    path_in_dir = os.listdir("../data")
    path_in_dir = [path for path in path_in_dir if category in path]
    if len(path_in_dir) == 0:
        cache_path = None
    else:
        cache_path = os.path.join("../data", max(path_in_dir))
    collaboration_graph = Collaboration_Graph_Scraper(anchor_author=scholar_name, category_ls=[category], max_depth=2, max_results=2000, save_path=save_path, cache_path=cache_path)
    max_workers = 4
    collaboration_graph.build_collaboration_graph_from_author(max_workers=max_workers)