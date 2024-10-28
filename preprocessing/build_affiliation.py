from utils import Affiliation_Builder

cache_path = '../data/author_affiliation/author_affiliation_parallel.pkl'
builder = Affiliation_Builder(from_cache=True, cache_path=cache_path)
builder(parallel=True, max_workers=8)  # Build affiliations