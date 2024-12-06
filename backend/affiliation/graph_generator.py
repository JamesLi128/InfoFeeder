from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
from utils import RecommendationLetterSeeker
from rapidfuzz import fuzz, process

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def group_queries(queries, threshold=90):
    """
    Groups similar queries using fuzzy matching.
    
    Args:
        queries (list of str): List of queries to be grouped.
        threshold (int): Similarity threshold for grouping (0-100).
        
    Returns:
        list of list: Groups of similar queries.
    """
    new_queries = set()
    # print(queries)
    for institution, departments in queries:
        new_queries.update(institution)
    queries = list(new_queries)
    groups = []
    used_indices = set()

    for i, query in enumerate(queries):
        if i in used_indices:
            continue

        for j, other_query in enumerate(queries):
            if j in used_indices:
                continue
            
            
    
    for i, query in enumerate(queries):
        if i in used_indices:
            continue
        
        # Start a new group
        group = [query]
        used_indices.add(i)
        
        for j, other_query in enumerate(queries):
            if j in used_indices:
                continue
            
            # Calculate similarity
            similarity = fuzz.ratio(query, other_query)
            if similarity >= threshold:
                group.append(other_query)
                used_indices.add(j)
        
        groups.append(group)

    institution2group_head = {}
    for group in groups:
        group_head = max(group, key=len)
        for institution in group:
            institution2group_head[institution] = group_head
    
    return institution2group_head

@app.route('/get_graph_data', methods=['POST'])
def get_graph_data():
    data = request.json
    scholar_name = data.get('scholarName')
    
    root_dir = os.listdir("../data")
    paper_categories = ["cs.AI", "cs.CC", "cs.LG", "math.CO", "math.OC", "stat.ML"]
    affiliation_path = '../data/author_affiliation/author_affiliation_parallel.pkl'
    
    result_dict = {}
    for paper_category in paper_categories:
        crnt_category_collaboration_path_ls = [path for path in root_dir if paper_category in path]
        most_recent_collaboration_path = max(crnt_category_collaboration_path_ls)
        most_recent_collaboration_path = os.path.join("../data", most_recent_collaboration_path)
        letter_seeker = RecommendationLetterSeeker(author_affiliation_path=affiliation_path, collaboration_graph_path=most_recent_collaboration_path)
        
        _, collaborators, affiliations = letter_seeker.from_scholar_name2affiliation(scholar_name=scholar_name)
        result_dict[paper_category] = (collaborators, affiliations)
    
    collaborator_ls, affiliation_ls = letter_seeker.combine_results(result_dict)

    institution2group_head = group_queries(affiliation_ls, threshold=80)

    print(institution2group_head)

    unavailable_institution = "N/A"
    json_dict = {
        "nodes" : [{"id" : scholar_name, "group" : 0}],
        "links" : [{"source" : scholar_name, "target" : unavailable_institution}]
    }
    group_heads = set(institution2group_head.values())
    for institution_name in group_heads:
        json_dict["nodes"].append({"id" : institution_name, "group" : 1})
        json_dict["links"].append({"source" : scholar_name, "target" : institution_name})

    for collaborator, (institution_ls, departments_ls) in zip(collaborator_ls, affiliation_ls):
        json_dict["nodes"].append({"id" : collaborator, "group" : 2})
        if len(institution_ls) == 0:
            json_dict["links"].append({"source" : unavailable_institution, "target" : collaborator})
        for institution in institution_ls:
            json_dict["links"].append({"source" : institution2group_head[institution], "target" : collaborator})
    
    return jsonify(json_dict)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)