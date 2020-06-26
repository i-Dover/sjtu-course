import warnings
import numpy as np
from utils import check_array, check_random_state, check_whether_fitted, euclidean_distance, \
            euclidean_distances


def auto_init(X, n_clusters, random_state, n_local_trials=None):
    """
    Initialize the cluster centers by a smart way.
    Args:
        X(ndarray): shape of (n_samples, n_features).
        n_clusters(int): The number of clusters.
        random_state: RandomState instance.
        n_local_trials(int): The number of seeding trials for each cluster.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    centers[0] = X[center_id]

    # Initialize list of closest distance and calculate current potential
    # (1, n_features) + (n_samples, n_features) -> (1, n_samples)
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, squared=True
    )
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # (m_features) + (n_samples, n_features) -> (m, n_samples)
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, squared=True
        )

        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidate_pot = distance_to_candidates.sum(axis=1)

        best_candidate = np.argmin(candidate_pot)
        current_pot = candidate_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X[best_candidate]

    return centers


def one_iter(X, centers_old, labels):
    """
    Update labels and centers for one iteration.

    Args:
        X(ndarray): shape of (n_samples, n_features)
        centers_old(ndarray): shape of (n_clusters, n_features)
        labels(ndarray): shape of (n_samples, )

    Returns:
        centers_new(ndarray): shape of (n_clusters, n_features).
        centers_shift(ndarray): shape of (n_clusters, )
            The distances between old_centers and new_centers.
    """
    n_samples, n_features = X.shape[0], X.shape[1]
    n_clusters = centers_old.shape[0]
    centers_new = np.zeros_like(centers_old)
    center_to_sample_index = [[] for _ in range(n_clusters)]  # Record each sample to its corresponding cluster label
    pairwise_distances = np.zeros((n_samples, n_clusters), dtype=X.dtype)  # distances matrix
    for i in range(n_samples):
        for j in range(n_clusters):
            pairwise_distances[i, j] = euclidean_distance(X[i], centers_old[j])

    for i in range(n_samples):
        min_sq_dist = pairwise_distances[i, 0]  # let the distance with the first cluster be now min distance.
        label = 0
        for j in range(1, n_clusters):
            sq_dist = pairwise_distances[i, j]
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j
        labels[i] = label
        center_to_sample_index[label].append(i)  # the i'th sample belong to the label'th cluster

    # Update the centers
    for c in range(n_clusters):
        if len(center_to_sample_index[c]) == 0:
            centers_new[c, :] = np.zeros(n_features, dtype=X.dtype)
        else:
            idx = np.array(center_to_sample_index[c])
            centers_new[c, :] = np.mean(X[idx, :], axis=0)
    # del center_to_sample_index

    centers_shift = np.sum(np.abs(centers_new-centers_old), axis=1)
    return centers_new, centers_shift


def get_score(X, centers, labels):
    """
    Sum of squared distance between each sample and its assigned center.
    This is a simple score for determining the performance of the clustering.
    """
    n_samples = X.shape[0]
    score = 0
    for i in range(n_samples):
        j = labels[i]
        sq_dist = euclidean_distance(X[i], centers[j])
        score += sq_dist
    return score


def init_centers(X, n_clusters=5, init='auto', random_state=None,
                 init_size=None):
    """
    Initialize the cluster centers.

    Args:
        X(ndarray): (n_samples, n_features)
        n_clusters(int):
        init(str): {'random', 'auto'}.
            'random': initialize the centers randomly.
            'auto': initialize the centers by a smart way.
        random_state(RandomState instance):
        init_size(int or None): sampling init_size samples to do the init.
    """
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    if init_size is not None and init_size < n_samples:
        if init_size < n_clusters:
            init_size = 3 * n_clusters
        init_indices = random_state.randint(0, n_samples, init_size)
        X = X[init_indices]
        n_samples = X.shape[0]

    if isinstance(init, str) and init == 'auto':
        centers = auto_init(X, n_clusters, random_state=random_state)
    elif isinstance(init, str) and init == 'random':
        seeds = random_state.permutation(n_samples)[:n_clusters]
        centers = X[seeds]
    elif hasattr(init, '__array__'):  # also can input the centers by user
        centers = np.array(init, dtype=X.dtype)
    else:
        raise ValueError("The init parameter should be 'auto' or 'random' or an ndarray, but got type: {}".
                         format(type(init)))

    _validate_center_shape(X, n_clusters, centers)
    return centers


def k_means_single(X, n_clusters, max_iter=300, init='auto',
                   random_state=None, tol=1e-4):
    """
    A single run of k-means.
    Args:
        X(ndarray): (n_samples, n_features)
        n_clusters(int): The number of clusters.
        max_iter(int): Maximum number of iteration to run.
        init(str | ndarray): {'auto', 'random', ndarray}, default='auto'
            Method for initialization:
            'auto': selects initial cluster centers in a smart way.
            'random': selects initial cluster centers randomly.
            ndarray: shape of (n_clusters, n_features), and gives the initial centers.
        random_state(RandomState instance):
        tol(float): default = 1e-4.
            Relative tolerance with regards to Frobenius norm of the difference
            in the cluster centers of two iterations to declare convergence.

    Returns:
        centers(ndarray): shape of (n_clusters, n_features)
            Centers found at the last iteration of k-means.
        labels(ndarray): shape of (n_samples, )
            labels[i] is the index of the center the i'th
            sample is closest to.
        score(float): sum of squared distance to the closest
            center for all samples in the training set.
    """
    random_state = check_random_state(random_state)
    centers = init_centers(X, n_clusters, init, random_state=random_state)
    n_samples = X.shape[0]

    labels = np.full(n_samples, -1, dtype=np.int32)

    for i in range(max_iter):
        centers_new, centers_shift = one_iter(X, centers, labels)

        center_shift_total = centers_shift.sum()
        if center_shift_total <= tol:
            print('Converged at iteration {}: center shift {} lower than tolerance {}'.
                  format(i, center_shift_total, tol))
            break

        centers = centers_new

    score = get_score(X, centers, labels)
    return labels, centers, score


def get_labels(X, centers):
    """
    Compute the labels and the score of the given samples and centers.
    """
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]
    pairwise_distances = np.zeros((n_samples, n_clusters), dtype=X.dtype)
    labels = np.full(n_samples, -1, dtype=np.int32)
    score = 0
    for i in range(n_samples):
        for j in range(n_clusters):
            pairwise_distances[i, j] = euclidean_distance(X[i], centers[j])

    for i in range(n_samples):
        min_sq_dist = pairwise_distances[i, 0]  # let the distance with the first cluster be now min distance.
        label = 0
        for j in range(1, n_clusters):
            sq_dist = pairwise_distances[i, j]
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j
        labels[i] = label
        score += min_sq_dist

    return labels, score


def _validate_center_shape(X, n_centers, centers):
    if not isinstance(centers, np.ndarray):
        raise ValueError('The type of the initial centers should be np.ndarray, but '
                         'got {}'.format(type(centers)))
    if len(centers) != n_centers:
        raise ValueError('The shape of the initial centers (%s) '
                         'does not match the number of clusters %i'
                         % (centers.shape, n_centers))
    if centers.shape[1] != X.shape[1]:
        raise ValueError('The number of features of the initial centers {} '
                         'does not match the number of features of the data {}'.
                         format(centers.shape[1], X.shape[1]))


class KMeans(object):
    def __init__(self, n_clusters=5, init='auto', n_init=10,
                 max_iter=300, tol=1e-4, random_state=None):
        """
        Args:
            n_init(int): Number of time the k-means will be run with
                different centroid seeds. Th final results will be best output
                of n_init consecutive runs in terms of score.
            n_clusters(int): The number of clusters.
            init(str | ndarray): The initialization method. Can be 'random', 'auto'
                or assign a initial ndarray.
            max_iter(int): The max iterations to run.
            tol(float): Relative tolerance with regards to Frobenius norm of the difference
                in the cluster centers of two consecutive iterations to declare
                convergence.
            random_state(RandomState instance): default is None
        """
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _check_test_data(self, X):
        X = check_array(X, dtype=[np.float64, np.float32])
        n_samples, n_features = X.shape
        expected_n_features = self.centers_.shape[1]
        if not expected_n_features == n_features:
            raise ValueError('Incorrect number of features. Got {} features, but expected {}'.
                             format(n_features, expected_n_features))
        return X

    def fit(self, X):
        random_state = check_random_state(self.random_state)

        n_init = self.n_init
        if self.n_init <= 0:
            raise ValueError('Invalid number of initializations.'
                             'n_init={} must be bigger than zero'.format(self.n_init))
        if self.max_iter <= 0:
            raise ValueError('Number of iterations should be a positive number, '
                             'but got {} instead.'.format(self.max_iter))
        X = check_array(X, dtype=[np.float64, np.float32])

        # verify the number of samples is larger than n_clusters
        n_samples, n_features = X.shape
        if n_samples < self.n_clusters:
            raise ValueError('n_samples={} should be >= n_clusters={}'.format(n_samples, self.n_clusters))

        # Validate init array
        init = self.init
        if hasattr(init, '__array__'):
            init = check_array(init, dtype=[np.float64, np.float32])
            _validate_center_shape(X, self.n_clusters, init)

            if self.n_init != 1:
                warnings.warn('Explicit initial center position passed, '
                              'performing only one init in k-means instead of n_init={}'.format(self.n_init),
                              RuntimeWarning)
                n_init = 1

        # subtract the mean of X for more accurate distance computation
        X_mean = X.mean(axis=0)
        X -= X_mean
        if hasattr(init, '__array__'):
            init -= X_mean

        best_labels, best_centers, best_score = None, None, None
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        for seed in seeds:
            labels, centers, score = k_means_single(
                X, self.n_clusters, max_iter=self.max_iter,
                init=init, tol=self.tol,
                random_state=seed,
            )
            if best_score is None or score < best_score:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_score = score
        X += X_mean
        best_centers += X_mean

        actual_clusters = len(set(best_labels))
        if actual_clusters < self.n_clusters:
            warnings.warn('Number of actual clusters ({}) found smaller than n_clusters ({}). '
                          'Possibly due to duplicate points in X'.format(actual_clusters, self.n_clusters),
                          RuntimeWarning)

        self.centers_ = best_centers
        self.labels_ = best_labels
        self.score_ = best_score
        return self

    def predict(self, X):
        check_whether_fitted(self)  # whether fit() is called

        X = self._check_test_data(X)

        return get_labels(X, self.centers_)[0]

    def score(self, X):
        check_whether_fitted(self)  # whether fit() is called

        X = self._check_test_data(X)

        return -get_labels(X, self.centers_)[1]



