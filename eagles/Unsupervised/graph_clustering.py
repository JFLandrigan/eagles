from igraph import Graph, plot
from eagles.data_sets import data_loader

def binarize_matrix(mat=None):



    return mat

def graph_clustering(df=None, ft_cols=[], method='weighted', thresh=.5, node_names=None):

    if len(ft_cols) == 0:
        ft_cols = [col for col in df.columns]

    # create correlation matrix for
    corr_matrix = df[ft_cols].T.corr()

    if method == 'binarized_isolated':
        adj_matrix = corr_matrix.copy(deep=True)
        adj_matrix[adj_matrix < thresh] = 0
        adj_matrix[adj_matrix >= thresh] = 1
    elif method == 'binarized_nonisolate':
        adj_matrix = corr_matrix.copy(deep=True)
        adj_matrix = binarize_matrix(mat=adj_matrix)
    elif method == 'weighted':
        adj_matrix = corr_matrix.values
    else:
        print('Method not supoorted')
        return

    graph = Graph.Weighted_Adjacency(adj_matrix.tolist(), mode="undirected", attr="weight", loops=False)
    Louvain = graph.community_multilevel(weights=graph.es['weight'], return_levels=False)
    Q = graph.modularity(Louvain, weights=graph.es['weight'])

    import networkx as nx
    import pyintergraph
    import matplotlib.pyplot as plt
    nx_graph = pyintergraph.igraph2nx(graph)
    nx.draw(nx_graph)

    # pos = nx.get_node_attributes(nx_graph, 'pos')
    # dmin = 1
    # ncenter = 0
    # for n in pos:
    #     x, y = pos[n]
    #     d = (x - 0.5) ** 2 + (y - 0.5) ** 2
    #     if d < dmin:
    #         ncenter = n
    #         dmin = d
    # p = dict(nx.single_source_shortest_path_length(nx_graph, ncenter))
    # plt.figure(figsize=(8, 8))
    # nx.draw_networkx_edges(nx_graph, pos, nodelist=[ncenter], alpha=0.4)
    # nx.draw_networkx_nodes(nx_graph, pos, nodelist=list(p.keys()),
    #                        node_size=80,
    #                        node_color=list(p.values()),
    #                        cmap=plt.cm.Reds_r)
    #
    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-0.05, 1.05)
    # plt.axis('off')
    # plt.show()

    print("The number of clusters found was: " + str(len(Louvain)))
    print('The modularity value is: ' + str(Q))

    if method == 'weighted':
        graph.es['weight'] = adj_matrix[adj_matrix.nonzero()]

    if node_names:
        graph.vs['label'] = node_names  # or a.index/a.columns

    # layout = graph.layout("kk")
    # plot(graph, layout=layout)


    return [graph, Louvain, Q]


df = data_loader.load_iris()
ft_cols = [col for col in df.columns if col != 'species']