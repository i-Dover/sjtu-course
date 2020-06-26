import numbers
import numpy as np
from inspect import isclass


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


def check_array(array, dtype='float64', min_samples=1, min_features=1):
    
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


def euclidean_distance(a, b):
    """
    Compute euclidean distance between two points.
    Returns: float.
    """
    squared = (a - b)**2
    ed = np.sqrt(np.sum(squared))
    return ed


def euclidean_distances(x, y, squared=False):
    """
    Compute euclidean distances between two sets of points.

    Args:
        x(ndarray): shape of (x_samples, n_features).
        y(ndarray): shape of (y_samples, n_features).
        squared(bool): Whether to return squared distances.
    Returns:
         distances(ndarray): shape of (x_samples, y_samples).
    """
    x_samples = x.shape[0]
    y_samples = y.shape[0]
    distances = np.zeros((x_samples, y_samples))
    for i in range(x_samples):
        for j in range(y_samples):
            distances[i, j] = euclidean_distance(x[i], y[j])
    return distances**2 if squared else distances