from utils import url_generator, Collaboration_Graph_Scraper, visualize_collaboration_graph_matplotlib
from datetime import datetime
category_ls = ['cs.CC']

save_path = f"../data/{'_'.join(category_ls)}_collaboration_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
collaboration_graph = Collaboration_Graph_Scraper(anchor_author='Amitabh Basu', category_ls=category_ls, max_depth=2, max_results=2000, save_path=save_path)
max_workers = 4
collaboration_graph.build_collaboration_graph_from_author(max_workers=max_workers)