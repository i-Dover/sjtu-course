import numpy as np

from utils import check_random_state, check_whether_fitted, check_array, _randomize, NetworkRecord
import networkx as nx


def get_indexed_partition(partitions, n_pass):
    """
    Returns the partition of the nodes at the given pass.
    Refer to [1], we use term 'pass' as each partition step.

    partitions are list of dictionary and partitions[i] is the
    partition of the graph nodes at pass i.

    Args:
        partitions(list[dict]): a list of partitions.
            partitions[i] is a dict, where the keys are node indexes,
            and the values are the communities of each node.
        n_pass(int): the pass number which belongs to [0, len(partitions)-1].
    Returns:
        partition(dict): A dict where keys are the nodes and the
        values are the community it belongs to.
    """
    partition = partitions[0].copy()
    for i in range(1, n_pass + 1):
        for node in partition.keys():
            community = partition[node]
            partition[node] = partitions[i][community]
    return partition


def get_last_partition(graph, init_partition=None, weight='weight',
                       random_state=None, tol=1e-7):
    """
    Compute the best partition of the graph nodes which maximizes the modularity
    using the Louvain Method.

    Args:
        graph(networkx.Graph): the networkx Graph instance.
        init_partition(dict): the algorithm will start using this partition
            of the nodes. It's a dict where keys are their nodes and values
            are the communities.
        weight(str):
        random_state(np.RandomState instance | None | int):
        tol(float):
    Returns:
        best_partition(dict): The partition dict, with communities numbered from 0
            to number of communities.
    """
    # get all partitions
    partitions = get_all_partitions(graph=graph,
                                    init_partition=init_partition,
                                    weight=weight,
                                    random_state=random_state,
                                    tol=tol)
    return get_indexed_partition(partitions, len(partitions) - 1)  # return the last partition


def get_all_partitions(graph, init_partition=None, weight='weight',
                       random_state=None, tol=1e-7):
    """
    Find communities in the graph and return the associated partitions.

    Args:
        see in :function: 'get_last_partition'.
    Returns:
        partitions(list[dict]): A list of partitions.
            partitions[i] is a dict, where keys are their nodes and
            values are the corresponding communities.
    """
    random_state = check_random_state(random_state)
    nr = NetworkRecord()  # network recorder
    partitions = list()

    current_graph = graph.copy()

    # special case, when there is no link
    # the best partition is each node in its community
    if graph.number_of_edges() == 0:
        partition = dict([])
        nodes = graph.nodes()
        for i in range(len(nodes)):
            node = nodes[i]
            partition[node] = i
        return [partition]

    # record the current graph and partition
    nr.record(current_graph, weight, init_partition)
    _one_pass(current_graph, nr, weight, random_state)  # do first pass
    new_modularity = _compute_modularity(nr)  # compute new modularity
    modularity = new_modularity
    partition = _reorder_community(nr.node2community)
    partitions.append(partition)
    current_graph = generate_next_graph(current_graph, partition, weight)  # the next new graph, which use communities as nodes
    nr.record(current_graph, weight)

    while True:
        _one_pass(current_graph, nr, weight, random_state)
        new_modularity = _compute_modularity(nr)
        if new_modularity - modularity < tol:  # if modularity increases smaller than tol
            break
        partition = _reorder_community(nr.node2community)
        modularity = new_modularity
        partitions.append(partition)
        current_graph = generate_next_graph(current_graph, partition, weight)
        nr.record(current_graph, weight)
    return partitions


def _one_pass(graph, record, weight, random_state, tol=1e-7):
    """
    Compute one pass of partition.
    """
    is_modified = True  # default assume the partition will be modified
    cur_modularity = _compute_modularity(record)
    new_modularity = cur_modularity

    while is_modified:
        cur_modularity = new_modularity
        is_modified = False

        for node in _randomize(graph.nodes(), random_state):
            pre_community = record.node2community[node]  # the node now belongs to pre_community
            node_degree_norm = record.ndegrees.get(node, 0.0) / (record.graph_weight * 2.0)
            neighbor_communities = _neighbor_communities(node, graph, record, weight)
            # if remove the node from its community, the cost
            remove_cost = -1.0 * neighbor_communities.get(pre_community, 0) + \
                        (record.cdegrees.get(pre_community, 0.0) - record.ndegrees.get(node, 0.0)) * node_degree_norm
            _remove_node(node, pre_community, neighbor_communities.get(pre_community, 0.0), record)
            best_community = pre_community
            best_increase = 0
            for community, kin in _randomize(neighbor_communities.items(), random_state):
                # the increase when the node add to the neighbor community
                # we use relative modularity increase to reduce complexity.
                increase = remove_cost + 1.0 * kin - record.cdegrees.get(community, 0.0) * node_degree_norm
                if increase > best_increase:
                    best_increase = increase
                    best_community = community
            _insert_node(node, best_community, neighbor_communities.get(best_community, 0.0), record)
            if best_community != pre_community:  # changed community
                is_modified = True
        new_modularity = _compute_modularity(record)
        if new_modularity - cur_modularity < tol:
            break


def generate_next_graph(graph, partition, weight='weight'):
    """
    Construct a new graph, where the nodes are the communities.

    Args:
        partition(dict): a dict where keys are nodes and values are the
            corresponding communities.
        graph(networkx.Graph):
        weight(str):
    Returns:
        new_graph(networkx.Graph): new graph, where the nodes are the communities.
    """
    new_graph = nx.Graph()
    new_graph.add_nodes_from(partition.values())  # communities as nodes

    for node1, node2, edge in graph.edges(data=True):
        edge_weight = edge.get(weight, 1)
        community1 = partition[node1]
        community2 = partition[node2]
        # get existed edge weight
        weight_precious = new_graph.get_edge_data(community1, community2, {weight: 0}).get(weight, 1)
        new_graph.add_edge(community1, community2, **{weight: weight_precious + edge_weight})

    return new_graph


def _reorder_community(dictionary):
    """
    Reorder the values of the dictionary from 0 to n.
    """
    c = 0
    res = dictionary.copy()
    new_values = dict([])

    for key in dictionary.keys():
        value = dictionary[key]
        new_value = new_values.get(value, -1)
        if new_value == -1:  # this is a new community
            new_values[value] = c
            new_value = c
            c += 1
        res[key] = new_value
    return res


def _neighbor_communities(node, graph, cache, weight):
    """
    Compute the communities in the neighborhood of node in
    the graph given with the decomposition node2community
    """
    weights = {}
    for neighbor, edges in graph[node].items():
        if neighbor != node:
            edge_weight = edges.get(weight, 1)
            neighbor_community = cache.node2community[neighbor]
            weights[neighbor_community] = weights.get(neighbor_community, 0) + edge_weight
    return weights


def _remove_node(node, community, weight, record):
    # change community degrees
    record.cdegrees[community] = (record.cdegrees.get(community, 0.0) - record.ndegrees.get(node, 0.0))
    # change community inter degrees
    record.inter_degrees[community] = float(record.inter_degrees.get(community, 0.0) -
                                            weight - record.node_loops.get(node, 0))
    record.node2community[node] = -1


def _insert_node(node, community, weight, record):
    record.node2community[node] = community  # change node community
    # change community degrees
    record.cdegrees[community] = (record.cdegrees.get(community, 0.0) + record.ndegrees.get(node, 0.0))
    # change community inter degrees
    record.inter_degrees[community] = float(record.inter_degrees.get(community, 0.0) +
                                            weight + record.node_loops.get(node, 0.0))


def _compute_modularity(record):
    """
    :math: sum^c(in_degree / m - (degree / 2m) ** 2).
    in_degree is the inter degree of each community.
    degree is the total degree of each community.
    """
    links = float(record.graph_weight)
    if links == 0:
        raise ValueError('A graph without link has an undefined modularity.')
    modularity = 0
    for community in set(record.node2community.values()):
        inter_degree = record.inter_degrees.get(community, 0.0)  # inter degree in the community
        degree = record.cdegrees.get(community, 0.0)  # the degree connected with the community
        if links > 0:
            modularity += inter_degree * 1.0 / links - ((degree / (2 * links)) ** 2)
    return modularity


class Louvain(object):
    """
    Args:
        n_partitions(str | int): if n_partitions is int, should designate a integer
            for partitions number. if n_partitions is str, should be 'best', which means
            get best partitions.
        weight(str): get edge weight.
        random_state(None | np.RandomState instance):
        tol(float): tolerance for new_modularity - modularity.
    """
    def __init__(self, n_partitions=5, weight='weight',
                 random_state=None, tol=1e-7):
        self.n_partitions = n_partitions
        self.weight = weight
        self.random_state = random_state
        self.tol = tol
        self.partition_ = None

    def fit(self, edges, nodes, init_partition=None):
        """
        Args:
            edges(np.ndarray): shape of (n_edges, 2) or (n_edges, 3).
            nodes(np.ndarray): shape of (n_nodes, )
            init_partition(dict): the key is node index and the value is
                corresponding community.
        """
        edges = check_array(edges, dtype=[np.int32])
        nodes = check_array(nodes, dtype=[np.int32], keep_1d=True)
        nodes = np.sort(np.unique(nodes))
        nodes_in_edges = np.sort(np.unique(edges))
        if not np.all(nodes == nodes_in_edges):
            raise ValueError('The node indexes should be the same between "nodes" and "edges"')
        self.n_edges_ = edges.shape[0]
        self.n_nodes_ = nodes.shape[0]
        if init_partition is not None:
            if not isinstance(init_partition, dict):
                raise TypeError("The type of 'init_partition' should be dict, but got {}".format(type(init_partition)))
            else:
                if len(init_partition.keys()) != self.n_nodes_:
                    raise ValueError("In the 'init_partition', each node should have a community.")
        graph = self._create_graph(edges, nodes)
        if isinstance(self.n_partitions, str) and self.n_partitions == 'best':
            partition = get_last_partition(graph, init_partition, weight=self.weight,
                                           random_state=self.random_state, tol=self.tol)
            self.partition_ = partition
            return self
        if isinstance(self.n_partitions, int):
            partitions = get_all_partitions(graph, init_partition, weight=self.weight,
                                            random_state=self.random_state, tol=self.tol)
            n_passes = len(partitions)
            for i in range(n_passes-1, -1, -1):
                partition = get_indexed_partition(partitions, i)
                communities = np.array(list(partition.values()))
                if abs(len(np.unique(communities)) - self.n_partitions) <= 2:
                    self.partition_ = partition
                    break
            if self.partition_ is None:
                raise RuntimeError('The algorithm did not get proper partition, please fit again')
            return self

    def predict(self, nodes):
        check_whether_fitted(self)
        nodes = check_array(nodes, dtype=[np.int32], keep_1d=True)
        max_node_idx = np.max(nodes)
        if max_node_idx > self.n_nodes_:
            raise ValueError('The nodes should be in [0, {}], but the maximum index is {}'.
                             format(self.n_nodes_, max_node_idx))
        communities = np.zeros((len(nodes), ), dtype=nodes.dtype)
        for i in range(len(nodes)):
            node = nodes[i]
            communities[i] = self.partition_[node]  # get community
        return communities

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
        graph.add_nodes_from(range(len(nodes)))
        for edge_i in edges:
            node1 = edge_i[0]
            node2 = edge_i[1]
            if with_weight:
                weight = edge_i[2]
                graph.add_edge(node1, node2, weight)
            else:
                graph.add_edge(node1, node2)
        return graph


