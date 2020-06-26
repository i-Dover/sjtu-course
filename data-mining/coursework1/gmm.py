import warnings
import numpy as np
from scipy import linalg
from k_means import KMeans
from scipy.special import logsumexp
from utils import check_random_state, check_array, check_whether_fitted


def _check_shape(array, array_shape, array_name):
    if array.shape != array_shape:
        raise ValueError('The parameter {} should have the shape of {}, but got {}'.
                         format(array_name, array_shape, array.shape))


def estimate_gaussian_covariances_full(probs, X, nk, means, eps):
    """
    Estimate the full covariances matrices. Used in the M-step.

    Args:
        probs(ndarray): shape of (n_samples, n_components)
        X(ndarray): shape of (n_samples, n_features)
        nk(ndarray): shape of (n_components, )
        means(ndarray): shape of (n_components, n_features)
        eps(float):

    Returns:
        covariances(ndarray): shape of (n_components, n_features, n_features)
            The covariances matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(probs[:, k] * diff.T, diff) / nk[k]  # (n_features, n_features)
        covariances[k].flat[::n_features + 1] += eps  # diagonal values + eps
    return covariances


def estimate_gaussian_parameters(X, probs, eps):
    """
    Estimate the Gaussian distribution parameters.

    Returns:
        nk(ndarray): shape of (n_components, )
            The number of data samples in the current components.
        means(ndarray): shape of (n_components, n_features)
            The centers of the current components
        covariances(ndarray): The covariance matrix of the current components
            The shape depends on the covariance_type.
    """
    nk = probs.sum(axis=0) + 10 * np.finfo(probs.dtype).eps
    means = np.dot(probs.T, X) / nk[:, np.newaxis]
    covariances = estimate_gaussian_covariances_full(probs, X, nk, means, eps)
    return nk, means, covariances


def compute_precision_cholesky(covariances):
    n_components, n_features, _ = covariances.shape
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(covariances):
        try:
            covariance_chol = linalg.cholesky(covariance, lower=True)
        except linalg.LinAlgError:
            raise ValueError('The cholesky decomposition failed.')
        precisions_chol[k] = linalg.solve_triangular(covariance_chol, np.eye(n_features), lower=True).T
    return precisions_chol


def compute_log_det(matrix, n_features):
    """
    Returns:
        log_det(ndarray): shape of (n_samples, )
            The det of the precision matrix for each component.
    """
    n_components, _, _ = matrix.shape
    log_det = (np.sum(np.log(matrix.reshape(n_components, -1)[:, ::n_features + 1]), 1))

    return log_det


def estimate_log_gaussian_prob(X, means, precisions):
    """
    Args:
        X(ndarray): shape of (n_samples, n_features)
        means(ndarray): shape of (n_components, n_features).
        precisions(ndarray):
    Returns:
        log_prob(ndarray): shape of (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    log_det = compute_log_det(
        precisions, n_features
    )

    log_prob = np.empty((n_samples, n_components))
    for k, (mu, precision) in enumerate(zip(means, precisions)):
        y = np.dot(X, precision) - np.dot(mu, precision)
        log_prob[:, k] = np.sum(np.square(y), axis=1)
    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


class GMM(object):
    """
    Gaussian Mixture Model.

    This class allows to estimate the parameters of a Gaussian mixture distribution.

    Args:
        n_components(int): The number of mixture components.
        tol(float): The convergence threshold.
        eps(float): Non-negative regularization added to the diagonal of covariance.
            Make sure that the covariance matrices are all positive.
        max_iter(int): The number of EM iteration to run.
        n_init(int): The number of runs to perform, and the best results are kept.
        init(str): {'kmeans', 'random'}.
            The method used to initialize the weights, the means and the precisions.
        weights_init(ndarray): shape of (n_components, )
            The initial weights that provided by user.
        means_init(ndarray): shape of (n_components, n_features)
            The initial means that provided by user.
        precisions_init(ndarray): shape of (n_components, n_features, n_features)
            The initial precisions that provided by user.
        random_state(RandomState instance):
    """
    def __init__(self, n_components=5, tol=1e-4,
                 eps=1e-6, max_iter=100, n_init=1, init='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None):
        self.n_components = n_components
        self.tol = tol
        self.eps = eps
        self.max_iter = max_iter
        self.n_init = n_init
        self.init = init
        self.random_state = random_state
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

    def _check_params(self, X):
        _, n_features = X.shape
        
        if self.weights_init is not None:
            self.weights_init = self._check_weights(self.weights_init, self.n_components)
        if self.means_init is not None:
            self.means_init = self._check_means(self.means_init, self.n_components, n_features)
        if self.precisions_init is not None:
            self.precisions_init = self._check_precisions(self.precisions_init, self.n_components, n_features)

    def _init(self, X, random_state):
        n_samples, _ = X.shape

        if self.init == 'kmeans':
            probs = np.zeros((n_samples, self.n_components))  # probability matrix
            # use k-means to get the initial labels for each sample
            labels = KMeans(n_clusters=self.n_components, n_init=1, random_state=random_state).fit(X).labels_
            probs[np.arange(n_samples), labels] = 1
        elif self.init == 'random':
            probs = random_state.rand(n_samples, self.n_components)  # randomly generate the probability matrix
            probs /= probs.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError('Unimplemented initialization method {}'.format(self.init))

        weights, means, covariances = estimate_gaussian_parameters(
            X, probs, self.eps
        )
        weights /= n_samples  # the component weights prior

        self.weights_ = (weights if self.weights_init is None else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_chol_ = compute_precision_cholesky(covariances)
        else:
            self.precisions_chol_ = np.array(
                [linalg.cholesky(precision, lower=True) for precision in self.precisions_init]
            )

    def fit(self, X):
        """
        Estimate model parameters using X.

        The method fits the model n_init times and keeps the parameters with
        the largest likelihood or lower bound.
        """
        X = check_array(X, dtype=[np.float64, np.float32], min_samples=2)
        self._check_params(X)
        random_state = check_random_state(self.random_state)
        self._init(X, random_state)

        max_lower_bound = -np.infty
        self.converged = False

        n_samples, _ = X.shape
        for i in range(self.n_init):
            lower_bound = -np.infty
            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound
                log_prob_norm, log_prob = self._e_step(X)
                self._m_step(X, log_prob)
                lower_bound = self._compute_lower_bound(
                    log_prob_norm
                )
                lower_bound_change = lower_bound - prev_lower_bound
                if abs(lower_bound_change) < self.tol:
                    self.converged = True
                    break

            # The EM steps is going to maximize the lower_bound
            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_params = self.get_params()
        if not self.converged:
            warnings.warn('The model did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol.')

        self.set_params(best_params)
        self.lower_bound_ = max_lower_bound

    def _e_step(self, X):
        log_prob_norm, log_prob = self._estimate_prob_reps(X)
        return np.mean(log_prob_norm), log_prob

    def _m_step(self, X, log_probs):
        """
        M step.
        Args:
            X(ndarray): (n_samples, n_features)
            log_probs(ndarray): (n_samples, n_components)
                Log of the posterior probabilities of each sample in X.
        """
        n_samples, _ = X.shape
        self.weights_, self.means_, self.covariances_ = (
            estimate_gaussian_parameters(X, np.exp(log_probs), self.eps)
        )
        self.weights_ /= n_samples
        self.precisions_chol_ = compute_precision_cholesky(self.covariances_)

    def score(self, X):
        check_whether_fitted(self)
        X = self._check_test_data(X)
        return logsumexp(self._estimate_weighted_log_prob(X), axis=1).mean()  

    def predict(self, X):
        """
        Predict the labels for the test data using trained model.

        Returns:
            labels(ndarray): (n_samples, )
        """
        check_whether_fitted(self)
        X = self._check_test_data(X)

        # use the weighted posterior probability to get labels
        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def predict_prob(self, X):
        check_whether_fitted(self)
        X = self._check_test_data(X)
        _, log_prob = self._estimate_prob_reps(X)  # get the posterior probability
        return np.exp(log_prob)

    def _estimate_weighted_log_prob(self, X):
        """
        Returns:
            weighted_log_prob(ndarray): shape of (n_samples, n_component).
        """
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    def _estimate_log_prob(self, X):
        """
        Returns:
            log_prob(ndarray): shape of (n_samples, n_components).
        """
        return estimate_log_gaussian_prob(
            X, self.means_, self.precisions_chol_
        )

    def _estimate_log_weights(self):
        """
        Returns:
            log_weight(ndarray): shape of (n_components, )
        """
        return np.log(self.weights_)

    def _estimate_prob_reps(self, X):
        """
        Estimate log probabilities and responsibilities for each sample.

        Returns:
            log_prob_norm(ndarray): (n_samples, )
            log_reps(ndarray): (n_samples, n_components)
        """
        weigthed_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weigthed_log_prob, axis=1)
        log_reps = weigthed_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_reps

    def _compute_lower_bound(self, log_prob_norm):
        return log_prob_norm

    def get_params(self):
        return self.weights_, self.means_, self.covariances_, self.precisions_chol_

    def set_params(self, parmas):
        self.weights_, self.means_, self.covariances_, self.precisions_chol_ = parmas
        _, n_features = self.means_.shape

        self.precisions_ = np.empty(self.precisions_chol_.shape)
        for k, precision in enumerate(self.precisions_chol_):
            self.precisions_[k] = np.dot(precision, precision.T)

    def _check_weights(self, weights, n_components):
        weights = check_array(weights, dtype=[np.float64, np.float32])
        _check_shape(weights, (n_components, ), 'weights')
        if any(np.less(weights, 0)) or any(np.greater(weights, 1)):
            raise ValueError('The weights should be in the [0, 1], but got min value: {} max value: {}'.
                             format(np.min(weights), np.max(weights)))
        if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
            raise ValueError('The weights should be normalized, but got sum(weights) = {}'.format(np.sum(weights)))
        return weights

    def _check_means(self, means, n_components, n_features):
        means = check_array(means, dtype=[np.float64, np.float32])
        _check_shape(means, (n_components, n_features), 'means')
        return means

    def _check_precisions(self, precisions, n_components, n_features):
        if not isinstance(precisions, np.ndarray):
            raise ValueError("Expected array's type should be np.ndarray, but got type {}".
                             format(type(precisions)))
        if precisions.dtype not in [np.float64, np.float32]:
            precisions = precisions.astype(np.float64)
        precisions_shape = (n_components, n_features, n_features)
        _check_shape(precisions, precisions_shape, 'precisions')

        for precision in precisions:
            if not (np.allclose(precision, precision.T) and np.all(linalg.eigvalsh(precision) > 0.)):
                raise ValueError("precision should be symmetric , positive-definite")

        return precisions

    def _check_test_data(self, X):
        X = check_array(X, dtype=[np.float64, np.float32])
        n_samples, n_features = X.shape
        expected_n_features = self.means_.shape[1]
        if not expected_n_features == n_features:
            raise ValueError('Incorrect number of features. Got {} features, but expected {}'.
                             format(n_features, expected_n_features))
        return X