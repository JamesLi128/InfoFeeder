import os
import networkx as nx
import pickle
import feedparser
import re

class Collaboration_Graph_Scraper:
    """
    Firstly read from cache graph if exists, otherwise start from scratch.
    Need to offer an anchor point, like a paper or an author or a category.
    max_depth, max_results per etc. can be configured
    if multiple max values are set, the lowest will be used.
    """
    def __init__(self, cache_path : str, 
                 save_path : str = None,
                 anchor_author : str = None, 
                 anchor_paper : str = None, 
                 anchor_category : str = None, 
                 max_depth : int = 3,
                 max_results : int = 200) -> None:
        self.cache_path = cache_path
        self.save_path = save_path
        self.anchor_author = anchor_author
        self.anchor_paper = anchor_paper
        self.anchor_category = anchor_category
        self.max_depth = max_depth
        self.max_results = max_results
        # self.max_collaborations = max_collaborations
        # self.max_authors = max_authors
        # self.max_papers = max_papers
        self._load_graph()

    def _load_graph(self) -> None:
        """
        Load the collaboration graph from the cache if it exists.
        """
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.graph, self.paper_record = pickle.load(f)
        else:
            self.graph = nx.Graph()

    def _author_re(self, author_name : str) -> re.Pattern:
        """
        Generate a regex pattern for the given author name.
        
        Args:
            author_name (str): The name of the author.
            
        Returns:
            re.Pattern: A compiled regex pattern.
        """
        split = author_name.split(' ')
        middle_str = 's*'.join(split)
        final_str = f"^{middle_str}$"
        return re.compile(final_str, re.IGNORECASE)
        
    def search_step(self, depth : int, unexplored_authors : list):
        """
        Perform a search step to find collaborators and papers.
        
        Args:
            depth (int): The current depth of the search.
        """
        # Implement the logic for searching collaborators and papers
        if depth >= self.max_depth:
            for author in unexplored_authors:
                # doublecheck if searched_author == author
                author_re = self._author_re(author_name=author)
                search_url = url_generator(author_ls=[author], max_results=self.max_results)
                feed = feedparser.parse(search_url)
                for entry in feed.entries:
                    for searched_author in entry.authors:
                        if author_re.match(searched_author.name):
                            paper_id = entry.id
                            if paper_id not in self.paper_record:
                                self.paper_record[paper_id] = entry.authors
            


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
    start = f"start={kwargs.get('start', 0)}"
    max_results = f"max_results={kwargs.get('max_results', 200)}"
    return f"http://export.arxiv.org/api/query?search_query={search_query}&start={start}&max_results={max_results}"

