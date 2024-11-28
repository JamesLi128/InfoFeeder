import os
from tqdm import tqdm
from utils import EmbeddingGenerator

generator = EmbeddingGenerator()

pdf_folder_path = '../data/pdf'

pdf_name_ls = os.listdir(pdf_folder_path)

for pdf_file_name in tqdm(pdf_name_ls, desc='Generating embeddings'):
    paper_name = pdf_file_name[:-4]
    generator(paper_name=paper_name)