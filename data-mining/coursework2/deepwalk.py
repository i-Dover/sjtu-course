import math
import numpy as np
import random
from tqdm import tqdm
import warnings
import string
import networkx as nx
import pandas as pd
from gensim.models import Word2Vec
from collections import defaultdict
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from utils import parallel_generate_walks
from utils import check_whether_fitted, check_array

warnings.filterwarnings('ignore')


class getGraph:
    """Converts edges from text file to adjacenct list.
    Args:
        edge_file(str):
            Path to the file where edges of web-graph are stored.
    """
    def __init__(self, edge_file):
        self.edge_file = edge_file

    def get_connections(self):
        """Reads the edges from the edge_file and save it in adjacency list.

        Returns:
            edges(collections.defaltdict[list]):
                Adjacency list containing information of connections in the graph.
        """
        edges = defaultdict(list)

        with open(self.edge_file, 'r') as e_file:
            edge_list = e_file.readlines()

        for edge in edge_list:
            from_, to_ = edge.split('\t')
            from_, to_ = int(from_), int(to_[:-1])
            edges[from_].append(to_)

        return edges


class PageRank:
    """PageRank of pages visualized as a graph.

    Contains function to calculate the rank of pages using the Google's
    PageRank Algorithm.

    Args:
        beta(float):
            Probability with which teleports will occur.
        edges(collections.defaltdict[list]):
            Adjacency list containing information of connections in web-graph.
        tol(float):
            A small value and total error in ranks should be less than tol.
        max_iterations(int):
            Maximum number of times to apply power iteration.
        node_num(int):
            Number of nodes in the web-graph
    """

    def __init__(self, beta, edges, tol, max_iterations, node_num):
        self.beta = beta
        self.edges = edges
        self.tol = tol
        self.node_num = node_num
        self.max_iterations = max_iterations

    def get_pagerank_score(self):
        """PageRank score of all nodes in the graph.

        Returns:
            final_rank_vector(np.ndarray): shape of (node_num, )
                Contains PageRank of each node in the graph.
        """
        final_rank_vector = np.zeros(self.node_num)
        initial_rank_vector = np.fromiter(
            [1 / self.node_num for _ in range(self.node_num)], dtype='float')

        iterations = 0
        diff = math.inf

        while iterations < self.max_iterations and diff > self.tol:
            new_rank_vector = np.zeros(self.node_num)
            for parent in self.edges:
                for child in self.edges[parent]:
                    new_rank_vector[child] += (initial_rank_vector[parent] /
                                               len(self.edges[parent]))

            leaked_rank = (1 - sum(new_rank_vector)) / self.node_num
            final_rank_vector = new_rank_vector + leaked_rank
            diff = sum(abs(final_rank_vector - initial_rank_vector))
            initial_rank_vector = final_rank_vector
            iterations += 1

        return final_rank_vector


def get_pagerank(edge_file, node_num, key_map, beta=0.85, tol=1e-6, max_iterations=20):
    """
    Returns:
        pagerank_score(dict): the key is the node index and the corresponding
            value is the pagerank score.
    """
    gg = getGraph(edge_file)
    edges = gg.get_connections()

    pr = PageRank(beta, edges, tol, max_iterations, node_num)
    pagerank_score = pr.get_pagerank_score()
    print('Computing PageRank score has finished..., Sum of PageRank score: {}'.
          format(sum(pagerank_score)))
    pagerank_map = dict()
    for i in range(node_num):
        key_node = key_map[i]
        pagerank_map[key_node] = pagerank_score[i]
    return pagerank_map


def get_random_walk(graph, node, path_length, pagerank_score=None):
    random_walk = [node]
    for i in range(path_length-1):
        neighbors = list(graph.neighbors(node))
        temp = list(set(neighbors) - set(random_walk))
        if len(temp) == 0:
            break

        if pagerank_score is None:  # totally random
            random_node = random.choice(temp)
            random_walk.append(random_node)
            node = random_node
        else:
            # use PageRank score to get the next random walk
            # if the PageRank score is larger, the higher chance
            # to walk
            temp_pagerank = []
            for t in temp:
                temp_pagerank.append(pagerank_score[t])  # get node t's pagerank
            sum_pagerank = sum(temp_pagerank)
            norm_pagerank = [p / sum_pagerank for p in temp_pagerank]  # normalize
            pagerank_cumsum = np.cumsum(norm_pagerank)
            random_val = random.random()
            bl = (random_val > pagerank_cumsum).tolist()
            choice = bl.index(False)
            random_node = temp[choice]
            random_walk.append(random_node)
            node = random_node
    return random_walk


def get_all_random_walks(graph, num_random=5, path_length=10, pagerank_score=None):
    all_nodes = list(graph.nodes())
    random_walks = []
    for node in tqdm(all_nodes):
        for i in range(num_random):  # every node will randomly walk 5 times
            random_walks.append(get_random_walk(graph, node, path_length, pagerank_score))
    print('Already get all random walks, Number of random walks: {}'.format(len(random_walks)))
    return random_walks


def generate_random_str(random_length=16):
    str_list = [random.choice(string.digits + string.ascii_letters) for _ in range(random_length)]
    random_str = ''.join(str_list)
    return random_str


class DeepWalk(object):
    """
    Args:
        model(gensim.Word2Vec | str): default is 'default' to use
            default defined model.
        random_policy(str): {'pagerank', 'random', 'bias_random'}
            - 'pagerank', use PageRank score to design the random policy.
            - 'random', totally randomly choose the random walk way.
        path_length(int): random walk length.
        num_random(int): The number of random walk ways of each node.
        pagerank_score(dict | None): PageRank score of each node or None.
        dimensions(int): The embedding dimensions.
        epochs(int): The Word2Vec model run times.
        workers(int): The parallel workers.
    """
    def __init__(self, model='default', random_policy='pagerank',
                 path_length=10, num_random=5, pagerank_score=None,
                 dimensions=64, epochs=20, workers=1):
        if not isinstance(model, str):
            self.model = model
        else:  # default model
#             self.model = Word2Vec(window=4, sg=1, hs=0,
#                                   negative=10, alpha=0.03,
#                                   min_alpha=0.0007,
#                                   seed=14)
            self.model = Word2Vec(window=10, min_count=1, batch_words=4,
                                  workers=workers, size=dimensions)
        self.path_length = path_length
        self.num_random = num_random
        self.pagerank_score = pagerank_score
        self.random_policy = random_policy
        self.epochs = epochs
        self.workers = workers
        self.dimensions = dimensions

    def _create_graph(self, edges, nodes):
        """
            Create graph from the edges matrix and nodes vectors.

            Args:
                edges(np.ndarray): shape of (n_edges, 2) or shape of (n_edges, 3).
                nodes(np.ndarray): shape of (n_nodes, ).
            Returns:
                graph(networkx.Graph): a networkx Graph instance.
            """
        with_weight = edges.shape[1] == 3  # the 3rd dimension is edge weight.
        graph = nx.Graph()
        graph.add_nodes_from(nodes.tolist())
        for edge_i in edges:
            node1 = edge_i[0]
            node2 = edge_i[1]
            if with_weight:
                weight = edge_i[2]
                graph.add_edge(node1, node2, weight)
            else:
                graph.add_edge(node1, node2)
        return graph

    def _key_map(self):
        key_map = dict([])
        while True:
            for i in range(self.n_nodes_):
                key_map[i] = generate_random_str(15)
            value = np.array(list(key_map.values()))
            if len(np.unique(value)) == self.n_nodes_:
                break
        return key_map

    def _convert_edges(self, edges):
        self.key_map = self._key_map()
        for i in range(len(edges)):
            edges[i, 0] = self.key_map[edges[i, 0]]
            edges[i, 1] = self.key_map[edges[i, 1]]
        nodes = np.unique(edges)
        return edges, nodes

    def fit(self, csv_path):
        df = pd.read_csv(csv_path)
        edges = df.values
        nodes = np.unique(edges)
        self.n_edges_ = edges.shape[0]
        self.n_nodes_ = nodes.shape[0]
        df['source'] = df['source'].astype('object')
        df['target'] = df['target'].astype('object')
        edges = df.values
        edges, nodes = self._convert_edges(edges)
        graph = self._create_graph(edges, nodes)
        print('Creating the graph has finished...')

        if self.random_policy == 'random':
            random_walks = get_all_random_walks(graph, num_random=self.num_random,
                                                path_length=self.path_length, pagerank_score=None)
        elif self.random_policy == 'bias_random':
            self._precompute_probabilities(graph)
            print('Begin to generate random walks...')
            random_walks = self._generate_walks()

            print('Begin to train the model...')
            self.model = Word2Vec(random_walks, window=10, min_count=1, batch_words=4,
                                  workers=self.workers, size=self.dimensions)
            print('Model training has finished...')
            return self
        else:
            pagerank_score = self.pagerank_score
            if pagerank_score is None:
                txt_path = csv_path.split('.')[0] + '.txt'
                pagerank_score = get_pagerank(txt_path, self.n_nodes_, self.key_map)
            random_walks = get_all_random_walks(graph, num_random=self.num_random,
                                                path_length=self.path_length, pagerank_score=pagerank_score)
        self.model.build_vocab(random_walks, progress_per=2)
        print('Begin to train the model...')
        self.model.train(random_walks, total_examples=self.model.corpus_count,
                         epochs=self.epochs, report_delay=1)
        print('Model training has finished...')
        return self

    def predict(self, nodes):
        check_whether_fitted(self)
        nodes = np.sort(nodes).tolist()
        terms = []
        for node in nodes:
            terms.append(self.key_map[node])
        X = self.model[terms]
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(X)
        communities = kmeans.labels_
        return communities

    def _precompute_probabilities(self, graph, weight_key='weight'):
        """
        Precomputes transition probabilities for each node.
        """
        self.FIRST_TRAVEL_KEY = 'first_travel_key'
        self.PROBABILITIES_KEY = 'probabilities'
        self.NEIGHBORS_KEY = 'neighbors'
        self.WEIGHT_KEY = 'weight'
        self.NUM_WALKS_KEY = 'num_walks'
        self.WALK_LENGTH_KEY = 'walk_length'
        self.P_KEY = 'p'
        self.Q_KEY = 'q'
        self.d_graph = defaultdict(dict)
        self.p = 1
        self.q = 1
        self.sampling_strategy = {}

        nodes_generator = tqdm(graph.nodes(), desc='Computing transition probabilities')

        for source in nodes_generator:
            # Init probabilities dict for first travel
            if self.PROBABILITIES_KEY not in self.d_graph[source]:
                self.d_graph[source][self.PROBABILITIES_KEY] = dict()

            for current_node in graph.neighbors(source):
                # Init probabilities dict
                if self.PROBABILITIES_KEY not in self.d_graph[current_node]:
                    self.d_graph[current_node][self.PROBABILITIES_KEY] = dict()

                unnormalized_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for destination in graph.neighbors(current_node):
                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q

                    if destination == source:  # Backwards probability
                        ss_weight = graph[current_node][destination].get(weight_key, 1) * 1 / p
                    elif destination in graph[source]:  # If the neighbor is connected to the source
                        ss_weight = graph[current_node][destination].get(weight_key, 1)
                    else:
                        ss_weight = graph[current_node][destination].get(weight_key, 1) * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    d_neighbors.append(destination)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                self.d_graph[current_node][self.PROBABILITIES_KEY][
                    source] = unnormalized_weights / unnormalized_weights.sum()
                # Save neighbors
                self.d_graph[current_node][self.NEIGHBORS_KEY] = d_neighbors

            # Calculate first_travel weights for source
            first_travel_weights = []

            for destination in graph.neighbors(source):
                first_travel_weights.append(graph[source][destination].get(weight_key, 1))

            first_travel_weights = np.array(first_travel_weights)
            self.d_graph[source][self.FIRST_TRAVEL_KEY] = first_travel_weights / first_travel_weights.sum()

    def _generate_walks(self):
        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_random), self.workers)

        walk_results = Parallel(n_jobs=self.workers, temp_folder=None, require=None)(
            delayed(parallel_generate_walks)(self.d_graph,
                                             self.path_length,
                                             len(num_walks),
                                             idx,
                                             self.sampling_strategy,
                                             self.NUM_WALKS_KEY,
                                             self.WALK_LENGTH_KEY,
                                             self.NEIGHBORS_KEY,
                                             self.PROBABILITIES_KEY,
                                             self.FIRST_TRAVEL_KEY) for
            idx, num_walks
            in enumerate(num_walks_lists, 1))

        random_walks = flatten(walk_results)
        print('Already get all random walks, Number of random walks: {}'.format(len(random_walks)))
        return random_walks
