from scrap_utils import url_generator, Collaboration_Graph_Scraper, visualize_collaboration_graph_matplotlib

category_ls = ['cs.LG', 'cs.AI', 'math.CO', 'stat.ML']

collaboration_graph = Collaboration_Graph_Scraper(anchor_author='Amitabh Basu', category_ls=category_ls, max_depth=2, max_results=20)
collaboration_graph.search_step(unexplored_authors=['Amitabh Basu'])
collaboration_graph.save_graph("../data/test_graph.pkl")
visualize_collaboration_graph_matplotlib(
    collaboration_graph.graph,
    figsize=(14, 10),
    node_color='lightgreen',
    node_size=700,
    edge_color='blue',
    font_size=14,
    font_weight='bold',
    edge_label_color='darkred',
    title="Scholar Collaboration Network",
    show_edge_labels=True,
    layout='spring',
    save_path='../data/test.png'  # Change to e.g., 'collaboration_graph.png' to save the figure
)