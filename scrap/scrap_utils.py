import os
import networkx as nx
import pickle
import feedparser
import re
import matplotlib.pyplot as plt
from tqdm import tqdm

import concurrent.futures
import threading
class Collaboration_Graph_Scraper:
    """
    Firstly read from cache graph if exists, otherwise start from scratch.
    Need to offer an anchor point, like a paper or an author or a category.
    max_depth, max_results per etc. can be configured
    if multiple max values are set, the lowest will be used.
    """
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
        self._load_graph()
        if anchor_author not in self.graph:
            self.graph.add_node(anchor_author)

    def _load_graph(self) -> None:
        """
        Load the collaboration graph from the cache if it exists.
        """
        if self.cache_path != None and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.graph, self.paper_record = pickle.load(f)
        else:
            self.graph = nx.Graph()
            self.paper_record = {}

    def _author_re(self, author_name : str) -> re.Pattern:
        """
        Generate a regex pattern for the given author name.
        
        Args:
            author_name (str): The name of the author.
            
        Returns:
            re.Pattern: A compiled regex pattern.
        """
        split = author_name.split(' ')
        middle_str = r'\s*'.join(split)
        final_str = f"^{middle_str}$"
        return re.compile(final_str, re.IGNORECASE)
        
    def search_step(self, unexplored_authors : list, depth : int = 0, func_id=0) -> None:
        print(f"current depth: {depth}, unexplored authors: {unexplored_authors}")
        """
        Perform a search step to find collaborators and papers.
        
        Args:
            depth (int): The current depth of the search.
        """
        # Implement the logic for searching collaborators and papers

        if depth > self.max_depth:
            return

        else:
            new_unexplored_authors = []
            for author in tqdm(unexplored_authors, desc=f"Depth {depth}, Func {func_id}"):
                author_re = self._author_re(author_name=author)
                search_url = url_generator(author_ls=[author], max_results=self.max_results, category_ls=self.category_ls)
                feed = feedparser.parse(search_url)
                for entry in feed.entries:
                    for searched_author in entry.authors:
                        # print(searched_author.name)
                        if author_re.match(searched_author.name) is not None:
                            paper_id = entry.id.split('/abs/')[-1]
                            if paper_id not in self.paper_record:
                                author_name_ls = [au_dict.name for au_dict in entry.authors if au_dict.name not in self.graph]
                                self.paper_record[paper_id] = author_name_ls
                                new_unexplored_authors.extend(author_name_ls)
                                for co_author in entry.authors:
                                    if co_author.name not in self.graph:
                                        self.graph.add_node(co_author.name)
                                    if self.graph.has_edge(author, co_author.name):
                                        self.graph[author][co_author.name]['weight'] += 1
                                    else:
                                        self.graph.add_edge(author, co_author.name)
                                        self.graph[author][co_author.name]['weight'] = 1
            
            self.search_step(depth=depth+1, unexplored_authors=new_unexplored_authors, func_id=func_id+1)


    def parallel_search_step(self, unexplored_authors: list, depth: int = 0, func_id : int=0) -> None:
        print(f"Starting search at depth: {depth}, number of authors: {len(unexplored_authors)}")

        if depth > self.max_depth:
            return

        new_unexplored_authors = []
        author_lock = threading.Lock()
        graph_lock = threading.Lock()
        paper_record_lock = threading.Lock()

        def process_author(author):
            nonlocal new_unexplored_authors
            author_re = self._author_re(author_name=author)
            search_url = url_generator(author_ls=[author], max_results=self.max_results, category_ls=self.category_ls)
            feed = feedparser.parse(search_url)
            local_new_authors = []

            for entry in feed.entries:
                for searched_author in entry.authors:
                    if author_re.match(searched_author.name) is not None:
                        paper_id = entry.id.split('/abs/')[-1]
                        
                        with paper_record_lock:
                            if paper_id in self.paper_record:
                                continue
                            author_name_ls = [au_dict.name for au_dict in entry.authors if au_dict.name not in self.graph]
                            self.paper_record[paper_id] = author_name_ls

                        local_new_authors.extend(author_name_ls)

                        with graph_lock:
                            for co_author in entry.authors:
                                co_name = co_author.name
                                if co_name not in self.graph:
                                    self.graph.add_node(co_name)
                                if self.graph.has_edge(author, co_name):
                                    self.graph[author][co_name]['weight'] += 1
                                else:
                                    self.graph.add_edge(author, co_name)
                                    self.graph[author][co_name]['weight'] = 1

            with author_lock:
                new_unexplored_authors.extend(local_new_authors)

        # Use ThreadPoolExecutor to process authors in parallel
        max_workers = min(8, len(unexplored_authors) or 1)  # Adjust the number of workers as needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Using tqdm to show progress
            futures = {executor.submit(process_author, author): author for author in unexplored_authors}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                            desc=f"Depth {depth}, Func {func_id}"):
                try:
                    future.result()
                except Exception as exc:
                    author = futures[future]
                    print(f'Author {author} generated an exception: {exc}')

        # Remove duplicates by converting to a set
        unique_new_authors = list(set(new_unexplored_authors))

        # Recursive call for the next depth level
        self.search_step(depth=depth + 1, unexplored_authors=unique_new_authors, func_id=func_id + 1)



    def save_graph(self, save_path : str = None) -> None:
        """
        Save the collaboration graph to the given path.
        
        Args:
            save_path (str): The path to save the graph.
        """
        if save_path is None:
            save_path = self.save_path
        with open(save_path, 'wb') as f:
            pickle.dump((self.graph, self.paper_record), f)
        print(f"Graph saved to {save_path}")



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
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=font_size,
        font_weight=font_weight
    )
    
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