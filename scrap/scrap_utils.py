import os

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
    return '+OR+'.join([f'au:"{name}"' for name in author_name_ls])

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
    return '+OR+'.join([f'cat:"{category}"' for category in category_ls])

def query_generator(**kwargs) -> str:
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
        
    return '+AND+'.join(queries)