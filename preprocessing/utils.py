import os
import networkx as nx
import pickle
import feedparser
import re
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import threading
import tiktoken
import fitz
import queue
from openai import OpenAI
import torch
import time
import tempfile

from requests import RequestException

class Collaboration_Graph_Scraper:
    def __init__(self, cache_path : str = None, 
                 save_path : str = '../data/collaboration_graph.pkl',
                 anchor_author : str = None, 
                 anchor_paper : str = None, 
                 anchor_category : str = None, 
                 category_ls : list = None,
                 max_depth : int = 3,
                 max_results : int = 200) -> None:
        self.cache_path = cache_path
        self.save_path = save_path
        self.anchor_author = anchor_author
        self.anchor_paper = anchor_paper
        self.anchor_category = anchor_category
        self.category_ls = category_ls
        self.max_depth = max_depth
        self.max_results = max_results
        # self.max_collaborations = max_collaborations
        # self.max_authors = max_authors
        # self.max_papers = max_papers
        self.authors_to_explore = queue.Queue()
        self.depth_queue = queue.Queue()
        self.num_updates = 0

        self._load_graph()
        if anchor_author not in self.graph:
            self.graph.add_node(anchor_author)
            self.authors_to_explore.put(anchor_author)
            self.depth_queue.put(0)
        
        self.processed_authors = self.fully_explored_authors.copy()

    def _load_graph(self) -> None:
        if self.cache_path != None and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.graph, self.paper_id2author_ls, self.author2paper_id, self.fully_explored_authors = pickle.load(f)
        else:
            self.graph = nx.Graph()
            self.paper_id2author_ls = {}
            self.author2paper_id = {}
            self.fully_explored_authors = set()

    def _author_re(self, author_name : str) -> re.Pattern:
        split = author_name.split(' ')
        middle_str = r'\s*'.join(split)
        final_str = f"^{middle_str}$"
        return re.compile(final_str, re.IGNORECASE)
    
    def _update_graph_and_author2paper_id(self, author_name : str, collaborator_name : str, paper_id : str) -> None:
        # update the nodes in the graph
        if author_name not in self.graph:
            self.graph.add_node(author_name)
        if collaborator_name not in self.graph:
            self.graph.add_node(collaborator_name)
        # update author2paper_id
        if author_name not in self.author2paper_id:
            self.author2paper_id[author_name] = set()
        if collaborator_name not in self.author2paper_id:
            self.author2paper_id[collaborator_name] = set()

        if paper_id not in self.author2paper_id[author_name]:
            self.author2paper_id[author_name].add(paper_id)
            if self.graph.has_edge(author_name, collaborator_name):
                self.graph[author_name][collaborator_name]['weight'] += 1
            else:
                self.graph.add_edge(author_name, collaborator_name)
                self.graph[author_name][collaborator_name]['weight'] = 1

        if paper_id not in self.author2paper_id[collaborator_name]:
            self.author2paper_id[collaborator_name].add(paper_id)
            if self.graph.has_edge(author_name, collaborator_name):
                self.graph[author_name][collaborator_name]['weight'] += 1
            else:
                self.graph.add_edge(author_name, collaborator_name)
                self.graph[author_name][collaborator_name]['weight'] = 1
    
    def author_search_step(self, author_name : str, lock : threading.Lock, depth : int) -> None:
        """
        Perform a search for papers by a given author and update the collaboration graph.

        This function queries the arXiv API for papers authored by `author_name`, processes the results to find
        co-authors, and updates the collaboration graph and related data structures. It uses a lock to ensure
        thread-safe operations.

        Args:
            author_name (str): The name of the author to search for.
            lock (threading.Lock): A threading lock to ensure thread-safe operations.
            depth (int): The current depth of the search, used to limit the search depth.

        Returns:
            None
        """
        # Wait for 1 second to avoid overwhelming the server
        time.sleep(1)
        
        if depth > self.max_depth:
            return
        # query generation and parsing
        arxiv_query_url = url_generator(author_ls=[author_name], max_results=self.max_results, category_ls=self.category_ls)
        feed = feedparser.parse(arxiv_query_url)
        crnt_author_re = self._author_re(author_name)
        with lock:
            for entry in feed.entries:
                paper_id = entry.id.split('/abs/')[-1]
                for searched_author in entry.authors:
                    # if current paper is indeed by the author desired, and the paper is not already processed
                    if crnt_author_re.match(searched_author.name) and paper_id not in self.paper_id2author_ls:
                        # get the list of unprocessed collaborators, excluding the author himself
                        new_collaborator_name_ls = [collab_dict.name for collab_dict in entry.authors if (collab_dict.name not in self.processed_authors) and (collab_dict.name != author_name)]
                        # get the list of all collaborators
                        all_collaborator_name_ls = [collab_dict.name for collab_dict in entry.authors]
                        # update the paper_id to author list mapping
                        self.paper_id2author_ls[paper_id] = all_collaborator_name_ls
                        for collaborator_name in new_collaborator_name_ls:
                            # push the new collaborators to the queue
                            if collaborator_name not in self.authors_to_explore.queue:
                                self.authors_to_explore.put(collaborator_name)
                            # push the corresponding depth
                            self.depth_queue.put(depth+1)
                            # update the graph nodes and edges
                            self._update_graph_and_author2paper_id(author_name, collaborator_name, paper_id)
            # mark the current author as fully explored
            self.fully_explored_authors.add(author_name)
            print(f"Author {author_name} fully explored.")
            # record the number of updates
            self.num_updates += 1

    def build_collaboration_graph_from_author(self, max_workers : int = 8, max_num_attemps : int = 5) -> None:
        start_time = time.time()
        lock = threading.Lock()
        print("Building collaboration graph...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            while (not self.authors_to_explore.empty()) or any([future.running() for future in futures]):
                crnt_num_attemps = 0
                while crnt_num_attemps < max_num_attemps:
                    if (self.num_updates + 1) % 20 == 0:
                        print(f"Number of updates: {self.num_updates}")
                        print(f"Number of authors to explore: {self.authors_to_explore.qsize()}")
                        self.save_graph(lock=lock)
                    try:
                        crnt_author = self.authors_to_explore.get(timeout=2)
                        crnt_depth = self.depth_queue.get()
                        if crnt_depth > self.max_depth:
                            break
                        self.processed_authors.add(crnt_author)
                        # print(f"Searching for {crnt_author} at depth {crnt_depth}")
                        future = executor.submit(self.author_search_step, crnt_author, lock, crnt_depth)
                        futures.append(future)
                        break

                    except queue.Empty:
                        pass
                    
                    except RequestException as e:
                        crnt_num_attemps += 1
                        print(f"Request Exception: {e}")
                        print(f"{crnt_num_attemps}/{max_num_attemps} attemps made on querying {crnt_author}")
                        time.sleep(2)
                        if crnt_num_attemps == max_num_attemps:
                            print(f"Max number of attemps reached for author {crnt_author}, unable to fully explore.")
                            break

            for future in futures:
                future.result()
        self.save_graph(lock=lock)
        end_time = time.time()
        print(f"Graph built in {end_time - start_time:.2f} seconds.")

    def save_graph(self, save_path: str = None, lock : threading.Lock = None) -> None:
        """
        Save the collaboration graph to a pickle file using atomic save to prevent file corruption.
        
        If no `save_path` is provided, it uses the default `self.save_path`. 
        The function first writes to a temporary file and then renames it to ensure the 
        integrity of the file and avoid simultaneous write issues.
        """
        if save_path is None:
            save_path = self.save_path

        # Use a lock to prevent multiple threads from writing to the file at the same time
        
        with lock:
            temp_file = None
            try:
                # Write to a temporary file
                with tempfile.NamedTemporaryFile('wb', delete=False) as temp_file:
                    pickle.dump((self.graph, self.paper_id2author_ls, self.author2paper_id, self.fully_explored_authors), temp_file)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())  # Ensure data is written to disk
                
                # Atomically rename the temporary file to the final destination
                os.replace(temp_file.name, save_path)
                print(f"Graph saved to {save_path}")
            except Exception as e:
                print(f"Failed to save graph: {e}")
                if temp_file:
                    try:
                        os.remove(temp_file.name)
                    except OSError:
                        pass

class EmbeddingGenerator:
    def __init__(self, model_name : str = 'text-embedding-3-small', root_path="../data/") -> None:
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.model_name = model_name
        self.root_path = root_path
        self.pdf_downloader = PDFDownloader(root_path=root_path)
        if self.model_name == "text-embedding-3-small":
            self.embed_dim = 1536
        else:
            raise ValueError(f"Unsupported model name: {model_name}. Choose from 'text-embedding-3-small'.")

    def has_embedded(self, paper_name : str) -> bool:
        path = f"../data/embeddings/{paper_name}.pt"
        if '/' in paper_name:
            Warning("Should Input Paper Name, not Paper ID, deprecated arxiv naming")
            paper_name = paper_id2paper_name(paper_name)
        if os.path.exists(path):
            print(f"Embedding for paper {paper_name} already exists.")
            return True
        return False
    
    def has_abstract_embedded(self, paper_name : str) -> bool:
        path = f"../data/embeddings/{paper_name}_abstract.pt"
        if '/' in paper_name:
            Warning("Should Input Paper Name, not Paper ID, deprecated arxiv naming")
            paper_name = paper_id2paper_name(paper_name)
        if os.path.exists(path):
            print(f"Embedding for abstract of paper {paper_name} already exists.")
            return True
        return False
    
    def text2abstract(self, text : str) -> str:
        # Define regex to match the abstract section
        abstract_start = re.search(r'\bAbstract\b[:\s]*', text, re.IGNORECASE)
        if abstract_start:
            start_pos = abstract_start.end()
            # Attempt to find the next section, like 'Introduction' or '1.'
            abstract_end = re.search(r'\bIntroduction\b|^\d+\.', text[start_pos:], re.IGNORECASE | re.MULTILINE)
            if abstract_end:
                end_pos = start_pos + abstract_end.start()
            else:
                end_pos = len(text)
            
            # Extract and clean the abstract text
            abstract = text[start_pos:end_pos].strip()
            return abstract
        else:
            return "Abstract not found."
    
    def paper_id2text(self, paper_id : str) -> str:
        """
        Retrieve the text content of a paper given its ID.
        
        Parameters:
        paper_id (str): The unique identifier of the paper.
        
        Returns:
        str: The text content of the paper.
        """
        path = f"../data/pdf/{paper_id.split('/')[-1]}.pdf"
        if not os.path.exists(path):
            self.pdf_downloader(paper_id)
        doc = fitz.open(path)
        pages = [doc[i] for i in range(len(doc))]
        text = "".join([page.get_text() for page in pages])
        return text

    def text2token_chunks(self, text: str, tokenizer: tiktoken.core.Encoding, max_tokens: int = 8192):
        """
        Truncate the input text to fit within the maximum token limit.
        
        Parameters:
        text (str): The input string to be truncated.
        max_tokens (int): The maximum number of tokens allowed for the embeddings (default 8192).
        
        Returns:
        List[str]: A list of truncated text chunks that fit within the token limit.
        """
        tokens = tokenizer.encode(text)
        
        # If the text fits within the limit, return it as a single chunk
        if len(tokens) <= max_tokens:
            return [text]
        
        truncated_chunks = [ tokenizer.decode(tokens[i:i+max_tokens]) for i in range(0, len(tokens), max_tokens) ]
        truncated_chunks.append(tokenizer.decode(tokens[len(truncated_chunks)*max_tokens:]))
        return truncated_chunks

    def token_chunks2embedding_chunks(self, token_chunks: list[str]):
        client = OpenAI()

        # Generate embeddings for each chunk
        embedding_chunks = torch.zeros((len(token_chunks), self.embed_dim))
        for i, chunk in enumerate(token_chunks):
            crnt_embedding = client.embeddings.create(model=self.model_name, input=chunk)
            embedding_chunks[i] = torch.tensor(crnt_embedding.data[0].embedding)
        
        return embedding_chunks

    def paper_name2embedding_chunks(self, paper_name : str, max_tokens : int = 8192):
        """
        Generate embedding chunks for a paper given its name.
        
        Parameters:
        paper_name (str): The name of the paper.
        max_tokens (int): The maximum number of tokens allowed for the embeddings (default 8192).
        
        Returns:
        List[torch.Tensor]: A list of embedding chunks for the paper.
        """
        text = self.paper_id2text(paper_name)
        if not self.has_abstract_embedded(paper_name=paper_name):
            abstract = self.text2abstract(text)
            token_chunks = self.text2token_chunks(abstract, self.tokenizer, max_tokens)
            embedding_chunks = self.token_chunks2embedding_chunks(token_chunks)
            save_path = f"../data/embeddings/{paper_name}_abstract.pt"
            torch.save(embedding_chunks, save_path)
            print(f"Abstract embeddings for paper {paper_name} generated and saved to {save_path}")
        if self.has_embedded(paper_name=paper_name):
            return torch.load(f"../data/embeddings/{paper_name}.pt", weights_only=True)
        token_chunks = self.text2token_chunks(text, self.tokenizer, max_tokens)
        embedding_chunks = self.token_chunks2embedding_chunks(token_chunks)
        save_path = f"../data/embeddings/{paper_name}.pt"
        torch.save(embedding_chunks, save_path)
        print(f"Embeddings for paper {paper_name} generated and saved to {save_path}")
        return embedding_chunks
    
    def __call__(self, paper_name : str, max_tokens : int = 8192):
        if '/' in paper_name:
            Warning("Should Input Paper Name, not Paper ID, deprecated arxiv naming triggered")
            paper_name = paper_id2paper_name(paper_name)
        return self.paper_name2embedding_chunks(paper_name, max_tokens)

class PDFDownloader:
    def __init__(self, root_path : str = '../data/') -> None:
        self.root_path = root_path

    def _download_pdf(self, paper_id: str = None) -> None:
        url = f"http://export.arxiv.org/pdf/{paper_id}"
        paper_name = paper_id2paper_name(paper_id)
        save_path = os.path.join(self.root_path, f"pdf/{paper_name}.pdf")
        
        # Check if the file already exists
        if os.path.exists(save_path):
            print(f"PDF for paper {paper_id} already exists.")
            return None
        
        try:
            # Request the PDF file from arXiv
            response = requests.get(url)
            if response.status_code == 200:
                # Save the PDF content to a file
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded {paper_id}")
            else:
                print(f"Failed to download PDF for paper {paper_id}, status code: {response.status_code}")
        except Exception as e:
            print(f"Error downloading PDF for paper {paper_id}: {e}")
        
        return None

    def _parallel_download(self, paper_ids: list, max_workers: int = 5, delay: float = 1.0):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Submit tasks to the executor with a delay between each submission
            for paper_id in paper_ids:
                futures.append(executor.submit(self.download_pdf, paper_id))
                time.sleep(delay)  # Delay between submissions to avoid overwhelming the server
                
            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    future.result()  # Raise any exception caught during the download
                except Exception as e:
                    print(f"Error downloading file: {e}")

    def __call__(self, paper_id: str = None, paper_id_ls: list[str] = None, max_workers: int = 5, delay: float = 1.0):
        if paper_id is not None:
            self._download_pdf(paper_id)
        if paper_id_ls is not None:
            self._parallel_download(paper_id_ls, max_workers, delay)




def paper_id2paper_name(paper_id : str) -> str:
    return paper_id.replace('/', '_')

def query_generator_author(author_name_ls : list) -> str:
    """
    Generate a query string for the given list of author names.
    
    Args:
        author_name_ls (list): A list of author names.
        
    Returns:
        str: A query string formatted for use in a search.
    """

    # Convert author names to a search-friendly format
    # surname + initial and lower case
    # Example "Amitahb Basu" -> "basu_a"
    for i, name in enumerate(author_name_ls):
        space_split = name.split(' ')
        if len(space_split) > 1:
            search_name = "_".join([space_split[-1], space_split[0][0]])
            author_name_ls[i] = search_name.lower()
    return '+OR+'.join([f'au:{name}' for name in author_name_ls])

def query_generator_title(title : str) -> str:
    """
    Generate a query string for the given title.
    
    Args:
        title (str): The title of the work.
        
    Returns:
        str: A query string formatted for use in a search.
    """
    return "ti:" + title.replace(' ', '+')

def query_generator_category(category_ls : list) -> str:
    """
    Generate a query string for the given list of categories.
    
    Args:
        category_ls (list): A list of categories.
        
    Returns:
        str: A query string formatted for use in a search.
    """
    return '+OR+'.join([f'cat:{category}' for category in category_ls])

def url_generator(**kwargs) -> str:
    """
    Generate a query string based on the provided keyword arguments.
    
    Args:
        **kwargs: Keyword arguments that can include 'author', 'title', and 'category'.
        
    Returns:
        str: A query string formatted for use in a search.
    """
    queries = []
    
    if 'author_ls' in kwargs:
        queries.append( '(' + query_generator_author(kwargs['author_ls']) + ')' )
        
    if 'title' in kwargs:
        queries.append( '(' + query_generator_title(kwargs['title']) + ')' ) 
        
    if 'category_ls' in kwargs:
        queries.append( '(' + query_generator_category(kwargs['category_ls']) + ')' )
        
    search_query = '+AND+'.join(queries)
    start = f"{kwargs.get('start', 0)}"
    max_results = f"{kwargs.get('max_results', 200)}"
    return f"http://export.arxiv.org/api/query?search_query={search_query}&start={start}&max_results={max_results}&sortBy=lastUpdatedDate&sortOrder=descending"


def visualize_collaboration_graph_matplotlib(
    G,
    figsize=(12, 8),
    node_color='skyblue',
    node_size=500,
    edge_color='gray',
    font_size=12,
    font_weight='bold',
    edge_label_color='red',
    title="Collaboration Graph",
    show_edge_labels=True,
    layout='spring',
    save_path=None
):
    """
    Visualizes a NetworkX collaboration graph using Matplotlib.
    
    Parameters:
    - G (networkx.Graph or networkx.MultiGraph):
        The input collaboration graph. If it's a MultiGraph, multiple edges are aggregated.
        
    - figsize (tuple, optional):
        Size of the matplotlib figure. Default is (12, 8).
        
    - node_color (str or list, optional):
        Color of the nodes. Default is 'skyblue'.
        
    - node_size (int or list, optional):
        Size of the nodes. Default is 500.
        
    - edge_color (str or list, optional):
        Color of the edges. Default is 'gray'.
        
    - font_size (int, optional):
        Font size of the node labels. Default is 12.
        
    - font_weight (str, optional):
        Font weight of the node labels. Default is 'bold'.
        
    - edge_label_color (str, optional):
        Color of the edge labels. Default is 'red'.
        
    - title (str, optional):
        Title of the graph. Default is "Collaboration Graph".
        
    - show_edge_labels (bool, optional):
        Whether to display edge labels indicating the number of collaborations. Default is True.
        
    - layout (str, optional):
        Layout algorithm to use. Options include 'spring', 'circular', 'kamada_kawai', 'shell'.
        Default is 'spring'.
        
    - save_path (str, optional):
        If provided, saves the visualization to the specified path. Supported formats include PNG, PDF, SVG, etc.
        Default is None (does not save).
    
    Returns:
    - None. Displays the graph using Matplotlib and optionally saves it to a file.
    """
    
    # Step 2: Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    else:
        raise ValueError(f"Unsupported layout: {layout}. Choose from 'spring', 'circular', 'kamada_kawai', 'shell'.")
    
    # Step 3: Draw the graph
    plt.figure(figsize=figsize)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_color,
        node_size=node_size,
        edgecolors='black',  # Adding edge color to nodes for better visibility
        linewidths=1
    )
    
    # Draw labels
    # nx.draw_networkx_labels(
    #     G,
    #     pos,
    #     font_size=font_size,
    #     font_weight=font_weight
    # )
    
    # Prepare edge widths based on weight
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(weights) if weights else 1  # Avoid division by zero
    # Normalize edge widths for better visualization
    normalized_weights = [2 + (w / max_weight) * 4 for w in weights]  # Edge widths between 2 and 6
    
    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        width=normalized_weights,
        edge_color=edge_color,
        alpha=0.6
    )
    
    # Optionally, draw edge labels
    if show_edge_labels:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        # Convert integer weights to string for labeling
        edge_labels = {k: f"{v}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_color=edge_label_color,
            font_size=10
        )
    
    # Set title
    plt.title(title, fontsize=16)
    
    # Remove axes
    plt.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, format=save_path.split('.')[-1], dpi=300)
        print(f"Graph visualization saved to '{save_path}'.")
    
    # Display the graph
    plt.show()