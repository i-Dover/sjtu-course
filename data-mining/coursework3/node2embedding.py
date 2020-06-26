import numpy as np
import random
from tqdm import tqdm
import warnings
import networkx as nx
import pandas as pd
from gensim.models import Word2Vec
from collections import defaultdict
from joblib import Parallel, delayed
from utils import parallel_generate_walks
from utils import check_whether_fitted
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
}


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


class NodeEmbedding(object):
    """
    Args:
        model(gensim.Word2Vec | str): default is 'default' to use
            default defined model.
        random_policy(str): {'random', 'node2vec'}
            - 'node2vec', use node2vec policy.
            - 'random', totally randomly choose the random walk way.
        path_length(int): random walk length.
        num_random(int): The number of random walk ways of each node.
        p(int): p factor in node2vec.
        q(int): q factor in node2vec.
        window_size(int): window parameter in Word2Vec.
        dimensions(int): embedding vector size.
        workers(int): number of parallel workers.
        epochs(int): number fo epochs to run Word2Vec.
    """

    def __init__(self, model='default', random_policy='node2vec',
                 path_length=10, num_random=5, p=1., q=1.,
                 window_size=10, dimensions=64, workers=4, epochs=20):
        if isinstance(model, Word2Vec):
            self.model = model
        else:  # default model
            if random_policy == 'random':
                self.model = Word2Vec(window=4, sg=1, hs=0,
                                      negative=10, alpha=0.03,
                                      min_alpha=0.0007,
                                      seed=14)
            else:
                self.model = Word2Vec(window=window_size, min_count=1, batch_words=4,
                                      workers=workers, size=dimensions, sg=1)
        self.path_length = path_length
        self.num_random = num_random
        self.random_policy = random_policy
        self.window_size = window_size
        self.workers = workers
        self.dimensions = dimensions
        self.epochs = epochs
        self.q = q
        self.p = p
        self.FIRST_TRAVEL_KEY = 'first_travel_key'
        self.PROBABILITIES_KEY = 'probabilities'
        self.NEIGHBORS_KEY = 'neighbors'
        self.WEIGHT_KEY = 'weight'
        self.NUM_WALKS_KEY = 'num_walks'
        self.WALK_LENGTH_KEY = 'walk_length'
        self.P_KEY = 'p'
        self.Q_KEY = 'q'
        self.d_graph = defaultdict(dict)
        self.sampling_strategy = {}

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

    def fit(self, edges, nodes, model_path=None):
        """
        Args:
            edges(np.ndarray): shape of (n_edges, 2).
            nodes(np.ndarray): shape of (n_nodes, )
        """
        self.edges = edges
        self.n_edges_ = edges.shape[0]
        self.n_nodes_ = nodes.shape[0]
        graph = self._create_graph(edges, nodes)
        print('Creating the graph has finished...')

        if self.random_policy == 'random':
            random_walks = get_all_random_walks(graph, num_random=self.num_random,
                                                path_length=self.path_length, pagerank_score=None)
            random_walks = [map(str, walk) for walk in random_walks]
            self.model.build_vocab(random_walks, progress_per=2)
            print('Begin to train the model...')
            self.model.train(random_walks, total_examples=self.model.corpus_count,
                             epochs=self.epochs, report_delay=1)
            print('Model training has finished...')
            return self
        else:
            if model_path is not None:
                self.model = Word2Vec.load(model_path)
                print('Load the pre-trained model...')
            else:
                self._precompute_probabilities(graph)
                print('Begin to generate random walks...')
                random_walks = self._generate_walks()
                # random_walks = [map(str, walk) for walk in random_walks]
                print('Begin to train the model...')
                self.model.build_vocab(random_walks, progress_per=2)
                self.model.train(random_walks, total_examples=self.model.corpus_count,
                             epochs=self.epochs, report_delay=1)
                print('Model training has finished...')
            return self

    def get_edge_features(self, edges, edge_fn=None):
        """
        Get the edge feature vectors.
        Args:
            edges(np.ndarray): shape of (n_edges, 2).
            edge_fn(function): the function to compute edge feature vector.
        Returns:
            edges_features(np.ndarray): shape of (n_edges, self.dimension)
        """
        n_edges = edges.shape[0]
        nodes = np.unique(edges)
        nodes = nodes.tolist()
        nodes = [str(node) for node in nodes]
        edges_features = np.zeros((n_edges, self.dimensions))
        x = self.model[nodes]
        nodes_to_vector = dict()  # {node_no: node_vector}
        for i in range(len(nodes)):
            nodes_to_vector[nodes[i]] = x[i]
        for i in range(n_edges):
            source_node = str(edges[i, 0])
            target_node = str(edges[i, 1])
            source_node_vector = nodes_to_vector[source_node]
            target_node_vector = nodes_to_vector[target_node]
            if edge_fn is not None:
                edges_features[i, :] = edge_fn(source_node_vector, target_node_vector)
            else:
                warnings.warn('There is no "edge_fn", '
                              'will use the default "hadamard" edge_fn to compute edge feature vector.')
                edge_fn = edge_functions['hadamard']
                edges_features[i, :] = edge_fn(source_node_vector, target_node_vector)

        return edges_features

    def predict(self, test_edges, neg_edges_path='data/neg_edges.csv', ):
        """
        Predict the link label.
        Args:
            test_edges(np.ndarray): shape of (n_edges, 2).
            neg_edges_path(str): negative edge file path.
        Returns:
            test_label(np.ndarray): link label in {0, 1}.
            label_probs(np.ndarray): link label probabilities.
        """
        check_whether_fitted(self)
        pos_edge_features = self.get_edge_features(self.edges, edge_fn=edge_functions['hadamard'])
        neg_edges = pd.read_csv(neg_edges_path)
        neg_edges = neg_edges.values[:50000]
        neg_edge_features = self.get_edge_features(neg_edges, edge_fn=edge_functions['hadamard'])
        train_pos_labels = np.full((pos_edge_features.shape[0],), 1)  # positive labels
        train_neg_labels = np.full((neg_edge_features.shape[0],), 0)  # negative labels
        train_labels = np.hstack([train_pos_labels, train_neg_labels])
        edges_features_train = np.vstack([pos_edge_features, neg_edge_features])
        # train classifier
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=20, alpha=1e-4,
                            solver='sgd', random_state=1,
                            learning_rate_init=.1)
        X_train, X_test, y_train, y_test = train_test_split(edges_features_train, train_labels)
        print('Begin to train the classifier...')
        mlp.fit(X_train, y_train)
        score = roc_auc_score(mlp.predict(X_test), y_test)
        print('Have done classifier training process, the train auc score: {}'.format(score))
        test_edge_features = self.get_edge_features(test_edges, edge_fn=edge_functions['hadamard'])  # get test edge features
        test_labels = mlp.predict(test_edge_features)
        label_probs = mlp.predict_proba(test_edge_features)
        return test_labels, label_probs

    def _precompute_probabilities(self, graph, weight_key='weight'):
        """
        Precomputes transition probabilities for each node.
        """
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