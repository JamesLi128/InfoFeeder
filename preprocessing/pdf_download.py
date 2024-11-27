import pickle
import os
from utils import PDFDownloader, paper_name2paper_id

_, paper_id2author_ls, _, _, _ = pickle.load(open('../data/cs.AI_collaboration_graph_20241028_214220.pkl', 'rb'))
pdf_dir = '../data/pdf/'
exist_pdf_names = os.listdir(pdf_dir)
exist_pdf_ids = set([paper_name2paper_id(pdf_name)[:-4] for pdf_name in exist_pdf_names]) 
print(f'Existing PDFs: {len(exist_pdf_ids)}')

downloader = PDFDownloader()

max_workers = 4
paper_id_set = set(paper_id2author_ls.keys())
new_paper_id_ls = list(paper_id_set - exist_pdf_ids)  # Get the new paper IDs that need to be downloaded
downloader(paper_id_ls=new_paper_id_ls, max_workers=max_workers)  # Download PDFs