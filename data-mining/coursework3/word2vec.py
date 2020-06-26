import heapq
from timeit import default_timer
from copy import deepcopy
from collections import defaultdict
import threading
import warnings
from queue import Queue, Empty
import numpy as np
from six import iteritems, itervalues, string_types
from six.moves import xrange
from word2vec_map import Word2VecMapping, Word
import random
from tqdm import tqdm
from scipy.special import expit


def train_batch_sg(model, sentences, alpha):
    """
    Update Skip-Gram model by training on a batch of sentences.
    Args:
        model: :class: Word2Vec
        sentences(list[list]): A batch of sentences.
        alpha(float): learning rate.
    Returns:
        int: Number of words in the vocabulary actually used for training.
    """
    result = 0
    for sentence in sentences:
        word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab and
                       model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32]
        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)
            start = max(0, pos - model.window + reduced_window)
            for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                if pos2 != pos:
                    train_sg_pair(
                        model, model.wv.index2word[word.index], word2.index, alpha
                    )
        result += len(word_vocabs)
    return result


def train_batch_cbow(model, sentences, alpha):
    result = 0
    for sentence in sentences:
        word_vocabs = [
            model.wv.vocab[w] for w in sentence if w in model.wv.vocab and
            model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32
        ]
        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
            word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
            l1 = np.sum(model.wv.layer1[word2_indices], axis=0)
            if word2_indices and model.cbow_mean:
                l1 /= len(word2_indices)
            train_cbow_pair(model, word, word2_indices, l1, alpha)
        result += len(word_vocabs)
    return result


def train_sg_pair(model, word, context_index, alpha, learn_vectors=True,
                  learn_hidden=True, context_vectors=None, context_locks=None):
    """
    Train the passed model instance on a word and its context, using the Skip-Gram algorithm.
    """
    if context_vectors is None:
        context_vectors = model.wv.vectors
    if context_locks is None:
        context_locks = model.nn.vectors_lockf
    if word not in model.wv.vocab:
        return
    predicted_word = model.wv.vocab[word]

    l1 = context_vectors[context_index]
    lock_factor = context_locks[context_index]

    neule = np.zeros(l1.shape)
    if model.hs:
        l2a = deepcopy(model.nn.layer1[predicted_word.point])
        prod_term = np.dot(l1, l2a.T)
        fa = expit(prod_term)
        ga = (1 - predicted_word.code - fa) * alpha
        if learn_hidden:
            model.nn.layer1[predicted_word.point] += np.outer(ga, l1)
        neule += np.dot(ga, l2a)

    if model.negative:
        # use this word + 'negative' other random words not from this sentence
        word_indices = [predicted_word.index]
        while len(word_indices) < model.negative + 1:
            w = model.vocab.cum_table.searchsorted(model.random.randint(model.vocab.cum_table[-1]))
            if w != predicted_word.index:
                word_indices.append(w)
        l2b = model.nn.layer1neg[word_indices]
        prod_term = np.dot(l1, l2b.T)
        fb = expit(prod_term)
        gb = (model.neg_labels - fb) * alpha
        if learn_hidden:
            model.nn.layer1neg[word_indices] += np.outer(gb, l1)
        neule += np.dot(gb, l2b)

    if learn_vectors:
        l1 += neule * lock_factor
    return neule


def train_cbow_pair(model, word, input_word_indices, l1, alpha, learn_vectors=True, learn_hidden=True,
                    context_vectors=None, context_locks=None):
    if context_vectors is None:
        context_vectors = model.wv.vectors
    if context_locks is None:
        context_locks = model.nn.vectors_lockf
    neule = np.zeros(l1.shape)
    if model.hs:
        l2a = model.wv.vectors[word.point]
        prod_term = np.dot(l1, l2a.T)
        fa = expit(prod_term)
        ga = (1. - word.code - fa) * alpha
        if learn_hidden:
            model.wv.vectors[word.point] += np.outer(ga, l1)
        neule += np.dot(ga, l2a)
    if model.negative:
        word_indices = [word.index]
        while len(word_indices) < model.negative + 1:
            w = model.vocab.cum_table.searchsorted(model.random.randint(model.vocab.cum_table[-1]))
            if w != word.index:
                word_indices.append(w)
        l2b = model.nn.layer1neg[word_indices]
        prod_term = np.dot(l1, l2b.T)
        fb = expit(prod_term)
        gb = (model.neg_labels - fb) * alpha
        if learn_hidden:
            model.nn.layer1neg[word_indices] += np.outer(gb, l1)
        neule += np.dot(gb, l2b)
    if learn_vectors:
        if not model.cbow_mean and input_word_indices:
            neule /= len(input_word_indices)
        for i in input_word_indices:
            context_vectors[i] += neule * context_locks[i]
    return neule


class BaseWordEmbeddingModel(object):
    def __init__(self, workers=3, vector_size=100, epochs=5,
                 batch_words=10000, sg=0, init_alpha=0.025, window=5, random_state=1, hs=0, negative=5,
                 ns_exponent=0.75, cbow_mean=1, min_alpha=0.0001, **kwargs):
        self.sg = int(sg)
        if vector_size % 4 != 0:
            warnings.warn('Consider setting layer size a multiple of 4 for greater performance.')
        self.vector_size = vector_size
        self.workers = workers
        self.batch_words = batch_words
        self.init_alpha = init_alpha
        self.min_alpha = float(min_alpha)
        self.random = np.random.RandomState(random_state)
        self.window = window
        self.epochs = epochs
        self.hs = int(hs)
        self.negative = int(negative)
        self.ns_exponent = ns_exponent
        self.cbow_mean = int(cbow_mean)
        self.running_training_loss = 0
        self.current_alpha = float(init_alpha)
        self.corpus_count = 0
        self.corpus_total_words = 0
        self.train_count = 0
        self.total_train_time = 0
        if self.negative > 0:
            self.neg_labels = np.zeros(self.negative + 1)
            self.neg_labels[0] = 1.

    def _set_train_params(self, **kwargs):
        raise NotImplementedError()

    def _check_training_params(self, epochs=None, total_sentences=None, total_words=None, **kwargs):
        if not self.wv.vocab:  # should be set by `build_vocab`
            raise RuntimeError("you must first build vocabulary before training the model")
        if not len(self.wv.vectors):
            raise RuntimeError("you must initialize vectors before training the model")

        if not hasattr(self, 'corpus_count'):
            raise ValueError(
                "The number of examples in the training corpus is missing. "
                "Please make sure this is set inside `build_vocab` function."
                "Call the `build_vocab` function before calling `train`."
            )

        if total_words is None and total_sentences is None:
            raise ValueError(
                "You must specify either total_examples or total_words, for proper job parameters updation"
                "and progress calculations. "
                "The usual value is total_examples=model.corpus_count."
            )
        if epochs is None:
            raise ValueError("You must specify an explict epochs count. The usual value is epochs=model.epochs.")

    def _train_one_batch(self, data_iterable, job_params, thread_private_mem):
        raise NotImplementedError()

    def _raw_word_count(self, job):
        return sum(len(sentence) for sentence in job)

    def _clear_post_train(self):
        raise NotImplementedError()

    def _log_epoch_end(self, ):
        raise NotImplementedError()

    def _log_train_end(self, raw_word_count, trained_word_count, total_elapsed):
        print('training on a {} raw words ({} effective words) took {}, {} effective words/s'.format(
            raw_word_count, trained_word_count, total_elapsed, trained_word_count / total_elapsed
        ))

    def _log_epoch_progress(self, progress_queue=None):
        sentence_count, trained_word_count, raw_word_count = 0, 0, 0
        job_tally = 0
        unfinished_worker_count = self.workers

        while unfinished_worker_count > 0:
            report = progress_queue.get()
            if report is None:
                unfinished_worker_count -= 1
                continue
            sentences, trained_words, raw_words = report
            job_tally += 1
            sentence_count += sentences
            trained_word_count += trained_words
            raw_word_count += raw_words
        return trained_word_count, raw_word_count, job_tally

    def _get_thread_working_mem(self):
        raise NotImplementedError()

    def _worker_loop(self, job_queue, progress_queue):
        """
        Train the model, lifting batches of data from the queue.

        Args:
            job_queue(Queue of (list of objects, (str, int))
                A queue of jobs still to be processed. The worker will take up jobs from this queue.
                The first element is the corpus chunk to be processed and the second is the dict of
                parameters.
            progress_queue(Queue of (int, int, int)
                A queue of progress reports. Each report is represented as a tuple of 3 elements.
                * Size of data chunk, for example number of sentences in the data chunk.
                * Effective word count used in training
                * Total word count used in training
        """
        thread_private_mem = self._get_thread_working_mem()
        jobs_processed = 0
        while True:
            job = job_queue.get()
            if job is None:
                progress_queue.put(None)
                break
            data_iterable, job_params = job
            tally, raw_tally = self._train_one_batch(data_iterable, job_params, thread_private_mem)
            progress_queue.put((len(data_iterable), tally, raw_tally))
            jobs_processed += 1

    def _job_producer(self, data_iterable, job_queue, cur_epoch=0, total_sentences=None, total_words=None):
        """
        Fill the jobs queue using the data found in the input stream.
        :param data_iterable:
        :param job_queue:
        :param cur_epoch:
        :param total_examples:
        :param total_words:
        :return:
        """
        job_batch, batch_size = [], 0
        pushed_words, pushed_examples = 0, 0
        next_job_params = self._get_current_alpha(cur_epoch)
        job_no = 0

        for data_idx, data in enumerate(data_iterable):
            data_length = self._raw_word_count([data])
            if batch_size + data_length <= self.batch_words:
                job_batch.append(data)
                batch_size += data_length
            else:
                job_no += 1
                job_queue.put((job_batch, next_job_params))
                if total_sentences:
                    pushed_examples += len(job_batch)
                    epoch_progress = 1.0 * pushed_examples / total_sentences
                else:
                    pushed_words += self._raw_word_count(job_batch)
                    epoch_progress = 1.0 * pushed_words / total_words
                next_job_params = self._get_next_alpha(next_job_params, epoch_progress, cur_epoch)
                job_batch, batch_size = [data], data_length
        if job_batch:
            job_no += 1
            job_queue.put((job_batch, next_job_params))
        for _ in range(self.workers):
            job_queue.put(None)

    def _train_epoch(self, data_iterable, cur_epoch=0, total_sentences=None, total_words=None,
                     queue_factor=2):
        print(len(data_iterable))
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)
        workers = [
            threading.Thread(
                target=self._worker_loop,
                args=(job_queue, progress_queue, )
            )
            for _ in range(self.workers)
        ]
        workers.append(threading.Thread(
            target=self._job_producer,
            args=(data_iterable, job_queue),
            kwargs={'cur_epoch': cur_epoch, 'total_sentences': total_sentences, 'total_words': total_words}
        ))
        for worker in workers:
            worker.daemon = True
            worker.start()
        trained_word_count, raw_word_count, job_tally = self._log_epoch_progress(
            progress_queue
        )
        return trained_word_count, raw_word_count, job_tally

    def train(self, sentences=None, total_sentences=None, total_words=None,
              epochs=None, init_alpha=None, min_alpha=None,
              word_count=0, queue_factor=2, **kwargs):
        """
        :param sentences:
        :param corpus_file:
        :param total_sentences:
        :param total_words:
        :param epochs:
        :param init_alpha:
        :param min_alpha:
        :param word_count:
        :param queue_factor:
        :param report_delay:
        :param compute_loss:
        :param kwargs:
        :return:
            tuple(int, int): Tuple of (effective word count after ignoring unknown words, total word count).
        """
        if sentences is None:
            raise ValueError("You must give the sentences to train.")
        self.running_training_loss = 0.0
        self._set_train_params(**kwargs)
        self.epochs = epochs
        self._check_training_params(
            epochs=epochs,
            total_sentences=total_sentences,
            total_words=total_words, **kwargs
        )
        trained_word_count = 0
        raw_word_count = 0
        start = default_timer() - 0.00001
        job_tally = 0
        for cur_epoch in range(self.epochs):
            print('Current epoch: {}'.format(cur_epoch))
            if sentences is not None:
                trained_word_count_epoch, raw_word_count_epoch, job_tally_epoch = self._train_epoch(
                    sentences, cur_epoch=cur_epoch, total_sentences=total_sentences, total_words=total_words,
                    queue_factor=queue_factor,
                )
            else:
                raise ValueError('You give the sentences to train.')
            trained_word_count += trained_word_count_epoch
            raw_word_count += raw_word_count_epoch
            job_tally += job_tally_epoch
        total_time = default_timer() - start
        self._log_train_end(raw_word_count, trained_word_count, total_time)
        self.train_count += 1  # number of times train() has been called
        self._clear_post_train()
        return trained_word_count, raw_word_count

    def _get_current_alpha(self, cur_epoch):
        """
        Get the learning rate used in the current epoch
        Args:
            cur_epoch(int): Current epoch.
        Returns:
            cur_alpha(float): The learning rate for this epoch (it is linearly reduced)
        """
        cur_alpha = self.init_alpha - ((self.init_alpha - self.min_alpha) * float(cur_epoch) / self.epochs)
        return cur_alpha

    def _get_next_alpha(self, job_params, epoch_done, cur_epoch):
        progress = (cur_epoch + epoch_done) / self.epochs
        next_alpha = self.init_alpha - (self.init_alpha - self.min_alpha) * progress
        next_alpha = max(self.min_alpha, next_alpha)
        self.current_alpha = next_alpha
        return next_alpha


def zeros_aligned(shape, dtype, order='c', align=128):
    nbytes = np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize
    buffer = np.zeros(nbytes + align, dtype=np.uint8)
    start_index = -buffer.ctypes.data % align
    return buffer[start_index:start_index + nbytes].view(dtype).reshape(shape, order=order)


class Word2Vec(BaseWordEmbeddingModel):
    """
    Attributes:
        wv(Word2VecVectors): The is object essentially contains the mapping between words
            and embeddings. After training, it can be used directly.
        vocab(Word2VecVocab): This object represents the vocabulary of the model
        nn(NN): This object represents the inner shallow neural network used to
            train the embeddings. The semantics of the network differ slightly in the
            two available training modes (CBOW or SG). The weights are then used as our
            embeddings
    """
    def __init__(self, dimension=100, init_alpha=0.025, min_alpha=0.0001,
                 window=5, min_count=5, sample=1e-3, random_state=1, workers=3,
                 sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1,
                 hashfxn=hash, epochs=5, sorted_vocab=0, batch_words=10000):
        """
        Args:
            dimension(int): The dimension of word vectors.
            window(int): Maximum distance between the current and predicted word within a sentence.
            min_count(int): Ignores all words with total frequency lower than this.
            workers(int): Use these many worker threads to train the model.
            sg(int): {0, 1} Training algorithm: 1 for Skip-Gram, otherwise CBOW.
            hs(int): {0, 1}. If 1, hierarchical softmax will be used for model training.
                If 0, and 'negative' is non-zero, negative sampling will be used.
            negative(int): If > 0, negative sampling will be used, the int for negative specifies
                how many 'noise words' should be drawn. If set to 0, no negative sampling is used.
            ns_exponent(float): The exponent used to shape the negative sampling distribution.
            cbow_mean(int): {0, 1}. If 0, use the sum of the context word vectors. If 1, use the mean
                only applies when cbow is used.
            init_alpha(float): The initial learning rate.
            random_state(int):
            hashfxn(function):
            epochs(int): Number of epcohs over the corpus.
            sorted_vocab(int): {0, 1}. If 1, sort the vocabulary by descending frequency.
            batch_words(int): Target size for batches of examples passed to worker threads.
        """
        self.wv = Word2VecMapping(dimension)
        self.vocab = Vocab(min_count=min_count, sample=sample, sorted_vocab=bool(sorted_vocab),
                           ns_exponent=ns_exponent)
        self.nn = NN(random_state=random_state, vector_size=dimension, hashfxn=hashfxn)
        super(Word2Vec, self).__init__(
            workers=workers, dimension=dimension,
            epochs=epochs, batch_words=batch_words, sg=sg,
            hs=hs, window=window, init_alpha=init_alpha, min_alpha=min_alpha, min_count=min_count,
            sample=sample, random_state=random_state, hashfxn=hashfxn, negative=negative, ns_exponent=ns_exponent,
            cbow_mean=cbow_mean
        )

    def _set_train_params(self, **kwargs):
        self.running_training_loss = 0

    def estimate_memory(self, vocab_size=None, report=None):
        """
        Estimate required memory for a model using current settings and provided vocabulary size.
        """
        vocab_size = vocab_size or len(self.wv.vocab)
        report = report or {}
        report['vocab'] = vocab_size * (700 if self.hs else 500)
        report['vectors'] = vocab_size * self.wv.vector_size * np.dtype(np.float32).itemsize
        if self.hs:
            report['layer1'] = vocab_size * self.nn.layer1_size * np.dtype(np.float32).itemsize
        if self.negative:
            report['layer1neg'] = vocab_size * self.nn.layer1_size * np.dtype(np.float32).itemsize
        report['total'] = sum(report.values())
        return report

    def _get_thread_working_mem(self):
        """
        Compute the memory used per worker thread.
        Returns:
            (np.ndarray, np.ndarray)
        """
        work = zeros_aligned(self.nn.layer1_size, dtype=np.float32)
        neul = zeros_aligned(self.nn.layer1_size, dtype=np.float32)
        return work, neul

    def build_vocab(self, sentences=None, update=False, **kwargs):
        """
        Build vocabulary from a sequence of sentences.
            sentences(list[list]):
            update(bool):
        """
        total_words, corpus_count = self.vocab.count_vocab(
            sentences=sentences
        )
        self.corpus_count = corpus_count
        self.corpus_total_words = total_words
        report_values = self.vocab.prepare_vocab(
            hs=self.hs, negative=self.negative, wv=self.wv, **kwargs
        )
        report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])
        self.nn.prepare_weights(self.hs, self.negative, self.wv, update=update)

    def _train_one_batch(self, sentences, alpha, inits):
        """
        Train the model on a single batch of sentences.

        Args:
            sentences(list[list]): Corpus chunk to be used in this training batch.
            alpha(float): The learning rate used in this batch.
            inits(np.ndarray, np.ndarray): Each worker threads private work memory.
        Returns:
            tuple(int, int): 2-tuple (effective word count after ignore unknown
                words and sentence length trimming, total word count).
        """
        tally = 0
        if self.sg:
            tally += train_batch_sg(self, sentences, alpha)
        else:
            tally += train_batch_cbow(self, sentences, alpha)
        return tally, self._raw_word_count(sentences)

    def _clear_post_train(self):
        """Remove all L2-normlized word vectors from the model"""
        self.wv.vectors_norm = None

    def _reset_running_loss(self):
        self.running_training_loss = 0

    def train(self, sentences=None, total_sentences=None, total_words=None,
              word_count=0, queue_factor=2, **kwargs):
        """
        Args:
            sentences(list[list]):
            total_sentences(int): Count of sentences.
            total_words(int): Count of raw words in sentences.
            word_count(int): Count of words already trained.
            queue_factor(int): Multiplier for size of queue (number of workers * queue_factor)
        """
        return super(Word2Vec, self).train(
            sentences=sentences, total_sentences=total_sentences, total_words=total_words,
            epochs=self.epochs, init_alpha=self.init_alpha, min_alpha=self.min_alpha,
            word_count=word_count, queue_factor=queue_factor
        )


def keep_vocab_item(word, count, min_count):
    default_res = count >= min_count
    return default_res


class Vocab(object):
    """
    Vocabulary Manager used by :class: 'Word2Vec'.
    """
    def __init__(self, min_count=5, sample=1e-3, sorted_vocab=True, ns_exponent=0.75):
        self.min_count = min_count
        self.sample = sample
        self.sorted_vocab = sorted_vocab
        self.ns_exponent = ns_exponent
        self.cum_table = None  # for negative sampling
        self.raw_table = None

    def count_vocab(self, sentences):
        sentence_no = -1
        total_words = 0
        vocab = defaultdict(int)
        for sentence_no, sentence in enumerate(sentences):
            for word in sentence:
                vocab[word] += 1
            total_words += len(sentence)
        corpus_count = sentence_no + 1
        self.raw_table = vocab
        return total_words, corpus_count

    def sort_vocab(self, wv):
        if len(wv.vectors):
            raise RuntimeError('cannot sort vocabulary after model weights already initialized.')
        wv.index2word.sort(key=lambda word: wv.vocab[word].count, reverse=True)
        for i, word in enumerate(wv.index2word):
            wv.vocab[word].index = i

    def prepare_vocab(self, wv, hs, negative, sample=None):
        sample = sample or self.sample
        drop_total = drop_unique = 0

        retain_total, retain_words = 0, []
        wv.index2word = []
        wv.vocab = {}
        for word, v in iteritems(self.raw_table):
            if keep_vocab_item(word, v, self.min_count):
                retain_words.append(word)
                retain_total += v
                wv.vocab[word] = Word(count=v, index=len(wv.index2word))
                wv.index2word.append(word)
            else:
                drop_unique += 1
                drop_total += v
        original_unique_total = len(retain_words) + drop_unique
        original_total = retain_total + drop_total

        if not sample:
            threshold_count = retain_total
        elif sample < 1.0:
            threshold_count = sample * retain_total
        else:
            threshold_count = int(sample * (3 + np.sqrt(5)) / 2)
        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = self.raw_table[w]
            word_probability = (np.sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            wv.vocab[w].sample_int = int(round(word_probability * 2**32))
        self.raw_vocab = defaultdict(int)
        report_values = {
            'drop_unique': drop_unique, 'retain_total': retain_total, 'downsample_unique': downsample_unique,
            'downsample_total': downsample_total, 'num_retained_words': len(retain_words)
        }
        if self.sorted_vocab:
            self.sort_vocab(wv)
        if hs:
            self.create_binary_tree(wv)
        if negative:
            self.make_cum_table(wv)
        return report_values

    def create_binary_tree(self, wv):
        heap = list(itervalues(wv.vocab))
        heapq.heapify(heap)
        for i in xrange(len(wv.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Word(count=min1.count * min2.count, index=i + len(wv.vocab), left=min1, right=min2))
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()
                if node.index < len(wv.vocab):
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    points = np.array(list(points) + [node.index - len(wv.vocab)], dtype=np.uint32)
                    stack.append((node.left, np.array(list(codes) + [0], dtype=np.uint8), points))
                    stack.append((node.right, np.array(list(codes) + [1], dtype=np.uint8), points))

    def make_cum_table(self, wv, domain=2**31 - 1):
        vocab_size = len(wv.index2word)
        self.cum_table = np.zeros(vocab_size, dtype=np.uint32)
        train_words_pow = 0.0
        for word_index in xrange(vocab_size):
            train_words_pow += wv.vocab[wv.index2word[word_index]].count**self.ns_exponent
        cumulative = 0.0
        for word_index in xrange(vocab_size):
            cumulative += wv.vocab[wv.index2word[word_index]].count**self.ns_exponent
            self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain


class NN(object):
    """
    Represents the inner shallow neural network used to train Word2Vec.
    """
    def __init__(self, vector_size=100, random_state=1, hashfxn=hash):
        self.hashfxn = hashfxn
        self.layer1_size = vector_size
        self.random_state = random_state

    def prepare_weights(self, hs, negative, wv, update=False):
        if not update:
            self.reset_weights(hs, negative, wv)
        else:
            self.update_weights(hs, negative, wv)

    def random_vector(self, seed_string, vector_size):
        random = np.random.RandomState(self.hashfxn(seed_string) & 0xffffffff)
        return (random.rand(vector_size) - 0.5) / self.layer1_size

    def reset_weights(self, hs, negative, wv):
        wv.vectors = np.empty((len(wv.vocab), wv.vector_size), dtype=np.float32)
        for i in xrange(len(wv.vocab)):
            wv.vectors[i] = self.random_vector(wv.index2word[i] + str(self.random_state), wv.vector_size)
        if hs:
            self.layer1 = np.zeros((len(wv.vocab), self.layer1_size), dtype=np.float32)
        if negative:
            self.layer1neg = np.zeros((len(wv.vocab), self.layer1_size), dtype=np.float32)
        wv.vectors_norm = None
        self.vectors_lockf = np.ones(len(wv.vocab), dtype=np.float32)

    def update_weights(self, hs, negative, wv):
        gained_vocab = len(wv.vocab) - len(wv.vectors)
        newvectors = np.empty((gained_vocab, wv.vector_size), dtype=np.float32)
        for i in xrange(len(wv.vectors), len(wv.vocab)):
            newvectors[i - len(wv.vectors)] = self.random_vector(wv.index2word[i] + str(self.random_state),
                                                                 wv.vector_size)
        if not len(wv.vectors):
            raise RuntimeError('')
        wv.vectors = np.vstack([wv.vectors, newvectors])
        if hs:
            self.layer1 = np.vstack([self.layer1, np.zeros((gained_vocab, self.layer1_size), dtype=np.float32)])
        if negative:
            pad = np.zeros((gained_vocab, self.layer1_size), dtype=np.float32)
            self.layer1neg = np.vstack([self.layer1, pad])
        wv.vector_norm = None
        self.vectors_lockf = np.ones(len(wv.vocab), dtype=np.float32)


def _create_graph(edges, nodes):
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


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import networkx as nx
    edges_path = 'data/course3_edge.csv'
    test_path = 'data/course3_test.csv'
    submission_path = 'data/course3_submissionSample.csv'
    edges = pd.read_csv(edges_path)
    edges = edges.values
    nodes = np.unique(edges)
    test_edges = pd.read_csv(test_path).values[:, 1:]
    test_nodes = np.unique(test_edges)
    nodes_all = np.unique(np.hstack([nodes, test_nodes]))  # get all nodes
    graph = _create_graph(edges, nodes)
    random_walks = get_all_random_walks(graph, num_random=5, path_length=10)
    new_random_walks = []
    for walk in random_walks:
        for i, node in enumerate(walk):
            walk[i] = str(node)
        new_random_walks.append(walk)
    word2vec = Word2Vec(workers=4, epochs=10, dimension=64, sg=1, window=10, min_count=1, batch_words=4, )
    word2vec.build_vocab(new_random_walks)
    word2vec.train(new_random_walks, total_sentences=word2vec.corpus_count)
