import os
import networkx as nx
import pickle
import feedparser
import re
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import tiktoken
import fitz
import queue
from openai import OpenAI
import torch
import time
import tempfile
import numpy as np
from pymupdf import FileDataError
from typing import Generator
from requests import RequestException
from datetime import datetime

import builtins

# Redefine print to include flush=True globally
original_print = print  # Keep a reference to the original print function

def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    original_print(*args, **kwargs)


class Collaboration_Graph_Scraper:
    def __init__(self, cache_path : str = None, 
                 save_path : str = '../data/collaboration_graph.pkl',
                 anchor_author : str = None, 
                 anchor_paper : str = None, 
                 anchor_category : str = None, 
                 category_ls : list = None,
                 max_depth : int = 3,
                 max_num_update : int = None,
                 max_results : int = 200) -> None:
        self.cache_path = cache_path
        self.save_path = save_path
        self.anchor_author = anchor_author
        self.anchor_paper = anchor_paper
        self.anchor_category = anchor_category
        self.category_ls = category_ls
        self.max_depth = max_depth
        self.max_num_update = max_num_update
        self.max_results = max_results
        self.profile = {
            "cache_path": cache_path,
            "save_path": save_path,
            "anchor_author": anchor_author,
            "anchor_paper": anchor_paper,
            "anchor_category": anchor_category,
            "category_ls": category_ls,
            "max_depth": max_depth,
            "max_num_update": max_num_update,
            "max_results": max_results
        }
        # self.max_collaborations = max_collaborations
        # self.max_authors = max_authors
        # self.max_papers = max_papers
        self.authors_to_explore = queue.Queue()
        self.depth_queue = queue.Queue()
        self.num_updates = 0
        self.num_finished = 0

        self._load_graph()
        if cache_path == None:
            if anchor_author not in self.graph:
                self.graph.add_node(anchor_author)
                self.authors_to_explore.put(anchor_author)
                self.depth_queue.put(0)
        else:
            leaf_authors = self._find_leaf_authors()
            for author in leaf_authors:
                self.authors_to_explore.put(author)
                self.depth_queue.put(0)
            if self.max_depth > 1 and self.max_num_update == None:
                Warning(f"Expanding graph out of {len(leaf_authors)} leaf authors, max_depth > 1, might take a long time, condider using smaller max_depth or setting max_num_update.")
        
        self.processed_authors = self.fully_explored_authors.copy()

    def _load_graph(self) -> None:
        if self.cache_path != None and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.graph, self.paper_id2author_ls, self.author2paper_id, self.fully_explored_authors, self.profile = pickle.load(f)
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
    
    def _find_leaf_authors(self) -> set:
        leaf_nodes = {node for node in self.graph.nodes() if self.graph.degree(node) == 1}
        degree_ls = [self.graph.degree(node) for node in self.graph.nodes()]
        # plt.hist(degree_ls, bins=100)
        # plt.title("Distribution of Node Degrees")
        # plt.show()
        return leaf_nodes
    
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
        time.sleep(np.random.uniform(0.5, 1.5))
        
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
            print(f"Author {author_name} fully explored at depth d+{depth}.")
            # record the number of updates
            self.num_finished += 1

    def expand_collaboration_graph(self, max_workers : int = 8, max_num_attemps : int = 5) -> None:
        try:
            assert(self.cache_path != None)
        except AssertionError:
            raise ValueError("Cache path not provided, unable to expand graph, try build_collaboration_graph_from_author instead.")
        start_time = time.time()
        lock = threading.Lock()
        print("Expanding collaboration graph...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            last_time = time.time()
            while (not self.authors_to_explore.empty()) or any([future.running() for future in futures]):
                crnt_num_attemps = 0
                if self.num_updates >= self.max_num_update:
                    while not self.authors_to_explore.empty():
                        self.authors_to_explore.get()
                        self.depth_queue.get()
                    num_unfinished_threads = sum([future.running() for future in futures])
                    print(f"Max number of updates reached, waiting for {num_unfinished_threads} threads to finish.")
                    time.sleep(10)
                    continue
                while crnt_num_attemps < max_num_attemps:
                    crnt_time = time.time()
                    time_ellapsed = crnt_time - last_time
                    if time_ellapsed > 5:
                        print(f"Number of updates: {self.num_updates}")
                        print(f"Number of authors to explore: {self.authors_to_explore.qsize()}")
                        self.save_graph(lock=lock)
                        last_time = time.time()
                    # if (self.num_finished + 1) % 20 == 0:
                    #     print(f"Number of updates: {self.num_updates}")
                    #     print(f"Number of authors to explore: {self.authors_to_explore.qsize()}")
                    #     self.save_graph(lock=lock)
                    try:
                        crnt_author = self.authors_to_explore.get(timeout=np.random.uniform(1.5, 2.5))
                        crnt_depth = self.depth_queue.get()
                        if crnt_depth > self.max_depth:
                            break
                        self.processed_authors.add(crnt_author)
                        # print(f"Searching for {crnt_author} at depth {crnt_depth}")
                        self.num_updates += 1
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
        print(f"Graph expanded in {end_time - start_time:.2f} seconds.")

    def build_collaboration_graph_from_author(self, max_workers : int = 8, max_num_attemps : int = 5) -> None:
        try:
            assert(self.anchor_author != None)
        except AssertionError:
            raise ValueError("Anchor author not provided, try expand_collaboration_graph instead.")
        start_time = time.time()
        lock = threading.Lock()
        print("Building collaboration graph...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            last_time = time.time()
            while (not self.authors_to_explore.empty()) or any([future.running() for future in futures]):
                crnt_num_attemps = 0
                while crnt_num_attemps < max_num_attemps:
                    crnt_time = time.time()
                    time_ellapsed = crnt_time - last_time
                    if time_ellapsed > 10:
                        print(f"Number of updates: {self.num_updates}")
                        print(f"Number of authors to explore: {self.authors_to_explore.qsize()}")
                        self.save_graph(lock=lock)
                        print(f"authors to explore {self.authors_to_explore.qsize()}, has running threads: {sum([future.running() for future in futures])}")
                        last_time = time.time()
                    # if (self.num_updates + 1) % 20 == 0:
                    #     print(f"Number of updates: {self.num_updates}")
                    #     print(f"Number of authors to explore: {self.authors_to_explore.qsize()}")
                    #     self.save_graph(lock=lock)
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
                        break
                    
                    except ConnectionResetError as e:
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
                    pickle.dump((self.graph, self.paper_id2author_ls, self.author2paper_id, self.fully_explored_authors, self.profile), temp_file)
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


class Affiliation_Builder:
    '''
    From downloaded papers, build a dictionary author2affiliation(departments and universities)
    '''
    def __init__(self, save_dir : str = '../data/author_affiliation/', 
                 pdf_dir : str = '../data/pdf/',
                 from_cache : bool = False, 
                 cache_path : str = None
                 ) -> None:
        if from_cache:
            assert( cache_path is not None and os.path.exists(cache_path) )
            with open(cache_path, 'rb') as f:
                self.processed_paper_name_set, self.author2affiliation = pickle.load(f)
        else:
            self.processed_paper_name_set = set()
            self.author2affiliation = {}
        self.num_paper_processed_at_start = len(self.processed_paper_name_set)
        self.save_dir = save_dir
        self.pdf_dir = pdf_dir
        self.client = OpenAI()
        self.assistant = self.client.beta.assistants.create(
            name = "Author-Institution Matcher",
            instructions = "Given a paper's first page text, identify the authors and their institutions. Organise response in the following format: for each row, [Author Name] | [Departments or Affiliations, divide by ',' if more than 1] | [University / College], put value N/A if nothing matches.",
            model = "gpt-4o-mini-2024-07-18"
        )

    def _exclude_processed_paper(self, paper_name_ls : list) -> list:
        paper_name_set = set(paper_name_ls) - self.processed_paper_name_set
        print( f"Found {len(paper_name_set)} / {len(paper_name_ls)} new papers to process.")
        return list(paper_name_set)

    def _yield_first_page_text(self): 
        paper_name_ls = self._exclude_processed_paper(os.listdir(self.pdf_dir))
        filtered_paper_name_ls = [name for name in paper_name_ls if name not in self.processed_paper_name_set]
        print(f"Found {len(filtered_paper_name_ls)} out of {len(paper_name_ls)} papers to process.")
        paper_path_ls = [os.path.join(self.pdf_dir, paper_name) for paper_name in filtered_paper_name_ls]
        for paper_path, paper_id in tqdm(zip(paper_path_ls, filtered_paper_name_ls), desc="Processing paper", total=len(filtered_paper_name_ls)):
            try:
                doc = fitz.open(paper_path)
                first_page = doc[0]
                yield first_page.get_text(), paper_id
            except FileDataError:
                print(f"Corruped file {paper_path}, skipping.")
                continue

    def _queue_first_page_text(self) -> queue.Queue:
        paper_name_ls = self._exclude_processed_paper(os.listdir(self.pdf_dir))
        filtered_paper_name_ls = [name for name in paper_name_ls if name not in self.processed_paper_name_set]
        print(f"Found {len(filtered_paper_name_ls)} out of {len(paper_name_ls)} papers to process.")
        paper_path_ls = [os.path.join(self.pdf_dir, paper_name) for paper_name in filtered_paper_name_ls]
        q = queue.Queue()
        for paper_path, paper_id in zip(paper_path_ls, filtered_paper_name_ls):
            try:
                doc = fitz.open(paper_path)
                first_page = doc[0]
                q.put((first_page.get_text(), paper_id))
            except FileDataError:
                print(f"Corruped file {paper_path}, skipping.")
                continue
        print( f"Queue built with {q.qsize()} / {len(paper_name_ls)} papers, {len(paper_name_ls) - q.qsize()} corruped.")
        return q

    def _extract_info(self, result : str) -> list:
        lines = result.strip().split('\n')
        matches = []
        for line in lines:
            author, departments, institution = line.split('|')
            matches.append((author.strip(), departments.strip(), institution.strip()))
        return matches
    
    def _build_affiliation_step(self, first_page: str, paper_name: str, lock : threading.Lock) -> None:
        random_sleep = np.random.uniform(0.5, 1)
        time.sleep(random_sleep)
        thread = self.client.beta.threads.create()
        message = self.client.beta.threads.messages.create( 
            thread_id=thread.id,
            role= "user",
            content=first_page
        )
        run = self.client.beta.threads.runs.create_and_poll( 
            thread_id=thread.id,
            assistant_id= self.assistant.id,
        )
        if run.status == "completed":
            messages = self.client.beta.threads.messages.list(
            thread_id=thread.id
            )
            content = messages.data[0].content[0].to_dict()['text']['value']
            matches = self._extract_info(content)
            with lock:
                for match in matches:
                    author, departments, institution = match
                    author = author.lower()
                    department_ls = [dept.strip() for dept in departments.split(',')]
                    if author not in self.author2affiliation:
                        self.author2affiliation[author] = {
                            institution : department_ls
                        }
                    else: 
                        if institution not in self.author2affiliation[author]:
                            self.author2affiliation[author][institution] = department_ls
                        else:
                            for dept in department_ls:
                                if dept not in self.author2affiliation[author][institution]:
                                    self.author2affiliation[author][institution].append(dept)
                self.processed_paper_name_set.add(paper_name)
        else:
            print(f"Failed to process the first page of paper {paper_name}.")

    def _build_affiliation_parallel(self, max_workers : int = 8) -> None:
        lock = threading.Lock()
        paper_queue = self._queue_first_page_text()
        total_num_papers = paper_queue.qsize()
        last_time = time.time()
        last_time_save = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            while not paper_queue.empty() or any([future.running() for future in futures]):
                crnt_time = time.time()
                time_elapsed = crnt_time - last_time
                time_elapsed_save = crnt_time - last_time_save
                if time_elapsed > 5:
                    print(f"Processed papers {len(self.processed_paper_name_set) - self.num_paper_processed_at_start}/{total_num_papers}")
                    last_time = time.time()
                
                if time_elapsed_save > 60:
                    self._save_affiliation()
                    last_time_save = time.time()

                try:
                    random_timeout = np.random.uniform(1, 2)
                    first_page, paper_id = paper_queue.get(timeout=random_timeout)
                    future = executor.submit(self._build_affiliation_step, first_page, paper_id, lock)
                    futures.append(future)
                except queue.Empty:
                    pass

    def _build_affiliation(self) -> None:
        for first_page, paper_id in self._yield_first_page_text():
            thread = self.client.beta.threads.create()
            message = self.client.beta.threads.messages.create( 
                thread_id=thread.id,
                role= "user",
                content=first_page
            )
            run = self.client.beta.threads.runs.create_and_poll( 
                thread_id=thread.id,
                assistant_id= self.assistant.id,
            )
            if run.status == "completed":
                messages = self.client.beta.threads.messages.list(
                thread_id=thread.id
                )
                content = messages.data[0].content[0].to_dict()['text']['value']
                matches = self._extract_info(content)
                for match in matches:
                    author, departments, institution = match
                    author = author.lower()
                    department_ls = [dept.strip() for dept in departments.split(',')]
                    if author not in self.author2affiliation:
                        self.author2affiliation[author] = {
                            institution : department_ls
                        }
                    else: 
                        if institution not in self.author2affiliation[author]:
                            self.author2affiliation[author][institution] = department_ls
                        else:
                            for dept in department_ls:
                                if dept not in self.author2affiliation[author][institution]:
                                    self.author2affiliation[author][institution].append(dept)
                self.processed_paper_name_set.add(paper_id)
            else:
                print(f"Failed to process the first page of paper {paper_id}, status: {run.status}.")

    def _save_affiliation(self, file_name : str) -> None:
        if save_path is None:
            save_path = os.path.join(self.save_dir, file_name)
        with open(save_path, 'wb') as f:
            pickle.dump((self.processed_paper_name_set, self.author2affiliation), f)
        print(f"Affiliation data saved to {save_path}.")

    def __call__(self, file_name : str = None, parallel : bool = False, max_workers : int = 8) -> None:
        if parallel:
            self._build_affiliation_parallel(max_workers=max_workers)
        else:
            self._build_affiliation()
        if file_name is None:
            if parallel:
                file_name = "author_affiliation_parallel.pkl"
            else:
                file_name = "author_affiliation.pkl"

        self._save_affiliation(file_name=file_name)

class PDFDownloader:
    def __init__(self, root_path : str = '../data/') -> None:
        self.root_path = root_path
        self.num_downloaded = 0
        self.num_failed = 0
        self.num_existed = 0

    def _download_pdf(self, paper_id: str = None, max_attemps : int = 5) -> None:
        url = f"http://export.arxiv.org/pdf/{paper_id}"
        paper_name = paper_id2paper_name(paper_id)
        save_path = os.path.join(self.root_path, f"pdf/{paper_name}.pdf")
        
        # Check if the file already exists
        if os.path.exists(save_path):
            print(f"PDF for paper {paper_id} already exists.")
            return None
        crnt_attemps = 0
        while crnt_attemps < max_attemps:
            if crnt_attemps > 0:
                print(f"Retrying download for {paper_id} (Attempt {crnt_attemps + 1} / {max_attemps})...")
                time.sleep(np.random.uniform(1, 3))
            try:
                # Request the PDF file from arXiv
                response = requests.get(url)
                if response.status_code == 200:
                    # Save the PDF content to a file
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Successfully downloaded {paper_id}")
                    return None
                else:
                    print(f"Failed to download PDF for paper {paper_id}, status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading PDF for paper {paper_id}: {e}")
                if e == 429:
                    print("Too many requests, sleeping for 60 seconds...")
                    time.sleep(60)
            crnt_attemps += 1
        
        return None
    
    def _download_pdf_step(self, paper_id: str = None, max_attemps : int = 5, lock : threading.Lock = None) -> None:
        print( f"Downloading PDF for paper {paper_id}...")
        url = f"http://export.arxiv.org/pdf/{paper_id}"
        paper_name = paper_id2paper_name(paper_id)
        save_path = os.path.join(self.root_path, f"pdf/{paper_name}.pdf")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
        }
        
        # Check if the file already exists
        if os.path.exists(save_path):
            print(f"PDF for paper {paper_id} already exists.")
            with lock:
                self.num_existed += 1
            return None
        crnt_attemps = 0
        while crnt_attemps < max_attemps:
            if crnt_attemps > 0:
                print(f"Retrying download for {paper_id} (Attempt {crnt_attemps + 1} / {max_attemps})...")
                time.sleep(2)
            try:
                # Request the PDF file from arXiv
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    # Save the PDF content to a file
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    with lock:
                        self.num_downloaded += 1
                    print(f"Successfully downloaded {paper_id}")
                    return None
                else:
                    print(f"Failed to download PDF for paper {paper_id}, status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading PDF for paper {paper_id}: {e}")
            crnt_attemps += 1
        with lock:
            print(f"Failed to download PDF for paper {paper_id} after {max_attemps} attempts.")
            self.num_failed += 1
        return None


    def _parallel_download(self, paper_ids: list, max_workers: int = 5, max_attemps: int = 5) -> None:
        lock = threading.Lock()
        paper_id_q = queue.Queue()
        for paper_id in paper_ids:
            paper_id_q.put(paper_id)
        last_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            while not paper_id_q.empty() or any([future.running() for future in futures]):
                crnt_time = time.time()
                time_elapsed = crnt_time - last_time
                if time_elapsed > 5:
                    print(f"{self.num_downloaded} PDFs downloaded so far, {self.num_failed} failed, {self.num_existed} already exist, in total {self.num_downloaded + self.num_failed + self.num_existed} / {len(paper_ids)} papers.")
                    last_time = time.time()
                
                try:
                    paper_id = paper_id_q.get(timeout=5)
                    future = executor.submit(self._download_pdf_step, paper_id, max_attemps, lock)
                    futures.append(future)
                except queue.Empty:
                    pass

            for future in futures:
                future.result()

        print(f"{self.num_downloaded} PDFs downloaded so far, {self.num_failed} failed, {self.num_existed} already exist, in total {self.num_downloaded + self.num_failed + self.num_existed} / {len(paper_ids)} papers.")
        

    def __call__(self, paper_id: str = None, paper_id_ls: list[str] = None, max_workers: int = 5, max_attemps: int = 5) -> None:
        if paper_id is not None:
            self._download_pdf(paper_id)
        if paper_id_ls is not None:
            self._parallel_download(paper_id_ls, max_workers, max_attemps)

def paper_name2paper_id(paper_name : str) -> str:
    return paper_name.replace('_', '/')

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