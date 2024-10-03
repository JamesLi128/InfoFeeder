import pickle
from utils import parallel_download

_, paper_id2author_ls, _, _ = pickle.load(open('../data/collaboration_graph.pkl', 'rb'))

max_num_download = 100
num_workers = 6
paper_id_ls = list(paper_id2author_ls.keys())
parallel_download(paper_id_ls[:max_num_download], num_workers, 2.0)