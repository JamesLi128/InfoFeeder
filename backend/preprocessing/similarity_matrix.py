import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import os

embedding_folder_path = '../data/embeddings'
embedding_name_ls = os.listdir(embedding_folder_path)
abstract_name_ls = [name for name in embedding_name_ls if 'abstract' in name]

embedding_matrix = torch.zeros((len(abstract_name_ls), 1536))
for i, embedding_name in enumerate(abstract_name_ls):
    embedding_matrix[i] = torch.load(os.path.join(embedding_folder_path, embedding_name), weights_only=True).mean(dim=0)

normalized_embedding_matrix = F.normalize(embedding_matrix, p=2, dim=1)

similarity_matrix = torch.mm(normalized_embedding_matrix, normalized_embedding_matrix.t())
argmin = torch.argmin(similarity_matrix[3])
print(abstract_name_ls[3])
print(abstract_name_ls[argmin])
sns.heatmap(similarity_matrix, cmap='viridis')
plt.title('Embedding Similarity Matrix')
plt.savefig('./img/embedding_similarity_matrix.png')
plt.show()