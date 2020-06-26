import numbers
import random
import numpy as np
from inspect import isclass
from tqdm import tqdm

def check_whether_fitted(estimator):
    if isclass(estimator):
        raise TypeError('{} is a class, not an instance.'.format(estimator))
    attrs = [v for v in vars(estimator) if v.endswith('_')]  # get the attributes of the instance

    msg = "This {} instance is not fitted yet. Please call 'fit' before using it."

    class NotFittedError(ValueError, AttributeError):
        pass

    if not attrs:
        raise NotFittedError(msg.format(type(estimator).__name__))


def check_random_state(seed):
    """
    Turn seed to np.random.RandomState instance.

    Args:
        seed(None | int | np.random.RandomState):
            If seed is None, return np.random.mtrand._rand
            If seed is int, return a new RandomState with seed.
            If seed is RandomState, return it.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomSate instance' % seed)


def check_array(array, dtype='float64', min_samples=1, min_features=1, keep_1d=False):
    dtype_default = isinstance(dtype, str) and dtype == 'float64'
    dtype_orig = getattr(array, 'dtype', None)
    if dtype_default:
        if dtype_orig is None:
            # default: use np.float64
            dtype = np.float64
        else:
            dtype = None
    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no need to convert dtype
            dtype = None
        else:
            # just select the first element of the list of accepted types.
            dtype = dtype[0]

    if not isinstance(array, np.ndarray):
        raise ValueError("Expected array's type should be np.ndarray, but got type {}".format(type(array)))
    if array.ndim != 2:
        if keep_1d:  # return the 1d vector
            array = np.array(array, dtype=dtype)
            return array
        raise ValueError('Expected 2D array, but got {}D array instead. '
                         'Please check the shape of your array, and try '
                         'to change it to the 2D array'.format(array.ndim))
    if min_samples > 0:
        n_samples = array.shape[0]
        if n_samples < min_samples:
            raise ValueError('Found array with {} sample(s), while a minimum of {} is required.'.
                             format(n_samples, min_samples))
    if min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < min_features:
            raise ValueError('Found array with {} feature(s), while a minimum of {} is required.'.
                             format(n_features, min_features))
    array = np.array(array, dtype=dtype)  # if the dtype need to change, we should create a new array
    return array


def _randomize(items, random_state):
    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items


def parallel_generate_walks(d_graph, global_walk_length, num_walks, cpu_num,
                            sampling_strategy=None, num_walks_key=None, walk_length_key=None,
                            neighbors_key=None, probabilities_key=None, first_travel_key=None):
    walks = list()
    pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):
        # Update progress bar
        pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:
            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:
                walk_options = d_graph[walk[-1]].get(neighbors_key, None)
                # Skip dead end nodes
                if not walk_options:
                    break

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                else:
                    probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]

                walk.append(walk_to)
            walk = list(map(str, walk))  # Convert all to strings
            walks.append(walk)

        pbar.close()
    return walks


class NetworkRecord:
    """
    Record some useful info about current network.
    Such as node degrees, community degrees, community inter degrees
    """
    def __init__(self):
        self.node2community = dict([])  # record each node's community
        self.graph_weight = 0    # total number of edges in the graph
        self.ndegrees = dict([])  # node degrees
        self.cdegrees = dict([])  # community degrees
        self.inter_degrees = dict([])  # community inter degrees
        self.node_loops = dict([])  # node loops

    def record(self, graph, weight, partition=None):
        """
        Record the useful info about the current graph.
        """
        self.node2community = dict([])
        self.ndegrees = dict([])
        self.cdegrees = dict([])
        self.inter_degrees = dict([])
        self.node_loops = dict([])
        self.graph_weight = graph.size(weight=weight)
        count = 0
        if partition is None:
            for node in graph.nodes():
                self.node2community[node] = count
                degree = float(graph.degree(node, weight=weight))
                if degree < 0:
                    raise ValueError('Node degree must be greater than zero, but got {}'.format(degree))
                self.cdegrees[count] = degree
                self.ndegrees[node] = degree
                edge = graph.get_edge_data(node, node, default={weight: 0})
                self.node_loops[node] = float(edge.get(weight, 1))
                self.inter_degrees[count] = self.node_loops[node]
                count += 1
        else:
            for node in graph.nodes():
                community = partition[node]
                self.node2community[node] = community
                degree = float(graph.degree(node, weight=weight))
                self.cdegrees[community] = self.cdegrees.get(community, 0) + degree  # add degree
                self.ndegrees[node] = degree
                increase = 0.0
                for neighbor, edge in graph[node].items():
                    edge_weight = edge.get(weight, 1)
                    if edge_weight <= 0:
                        raise ValueError('Edge weight must be greater than zero, but got {}'.format(edge_weight))
                    if partition[neighbor] == community:
                        if neighbor == node:
                            increase += float(edge_weight)
                        else:
                            increase += float(edge_weight) / 2.0
                self.inter_degrees[community] = self.inter_degrees.get(community, 0) + increase



