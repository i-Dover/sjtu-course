from six import string_types, integer_types
import numpy as np


class Word(object):
    """
    A single word item.
    """

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):
        return self.count < other.count

    def __str__(self):
        vals = ['{}:{}'.format(key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class Word2VecMapping(object):
    """
    Mapping from words to vectors for Word2Vec
    """
    def __init__(self, vector_size):
        self.vectors = np.zeros((0, vector_size))
        self.vocab = {}
        self.vector_size = vector_size
        self.vectors_norm = None
        self.index2word = []

    def __contains__(self, word):
        return word in self.vocab

    def add(self, words, weights, replace=False):
        if isinstance(words, string_types):
            words = [words]
            weights = np.array(weights).reshape(1, -1)
        elif isinstance(weights, list):
            weights = np.array(weights)
        in_vocab_mask = np.zeros(len(words), dtype=np.bool)
        for idx, word in enumerate(words):
            if word in self.vocab:
                in_vocab_mask[idx] = True
        for idx in np.nonzero(~in_vocab_mask)[0]:
            word = words[idx]
            self.vocab[word] = Word(index=len(self.vocab), count=1)
            self.index2word.append(word)
        self.vectors = np.vstack((self.vectors, weights[~in_vocab_mask]))
        if replace:
            in_vocab_idxs = [self.vocab[words[idx]].index for idx in np.nonzero(in_vocab_mask)[0]]
            self.vectors[in_vocab_idxs] = weights[in_vocab_mask]

    def __setitem__(self, words, weights):
        if not isinstance(words, list):
            words = [words]
            weights = weights.reshape(1, -1)
        self.add(words, weights, replace=True)

    def __getitem__(self, words):
        if isinstance(words, string_types):
            return self.get_vector(words)

        return np.vstack([self.get_vector(word) for word in words])

    def word_vec(self, word, use_norm=False):
        """
        Get 'word' representations in vector space,
        Args:
            word(str): Input word.
            use_norm(bool): If true, resulting vector will be L2-normalized.
        """
        if word in self.vocab:
            if use_norm:
                result = self.vectors_norm[self.vocab[word].index]
            else:
                result = self.vectors[self.vocab[word].index]
            result.setflags(write=False)
            return result
        else:
            raise KeyError("word {} not in vocabulary".format(word))

    def get_vector(self, word):
        return self.word_vec(word)

    def distances(self, w1):
        pass

    def words_closer_than(self, w1, w2):
        """
        Get all words that are closer to 'w1' than 'w2' is to 'w1'.
        Returns:
            list(str): List of words.
        """
        all_distances = self.distances(w1)
        w1_index = self.vocab[w1].index
        w2_index = self.vocab[w2].index
        closer_node_indices = np.where(all_distances < all_distances[w2_index])[0]
        return [self.index2word[index] for index in closer_node_indices if index != w1_index]

    def rank(self, w1, w2):
        return len(self.words_closer_than(w1, w2)) + 1