#coding:utf-8
from typing import Union, Tuple, List, Iterable, Hashable, Callable, TypeVar, Sequence as Seq, Optional as Opt
from numbers import Number
from random import sample
from itertools import chain, repeat

from numpy import ndarray, finfo, abs as np_abs
from torch import from_numpy
from torch import stack
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import HalfTensor, FloatTensor, DoubleTensor, ByteTensor, CharTensor, ShortTensor, IntTensor, LongTensor, _TensorBase
from .util import batched, get_torch_num_workers, get_default_int_size


#####################################################################
# Type aliases                                                      #
#####################################################################

T1 = TypeVar('T1')
T2 = TypeVar('T2')

FloatTensorTypes = (FloatTensor, HalfTensor, DoubleTensor)
FloatTensorType = Union[FloatTensorTypes]

IntTensorTypes = (ByteTensor, CharTensor, ShortTensor, IntTensor, LongTensor)
IntTensorType = Union[IntTensorTypes]

NumericTensorTypes = (*FloatTensorTypes, *IntTensorTypes)
TensorType = Union[NumericTensorTypes]

Numeric = Union[Number, TensorType, Variable, ndarray]
ArrayTypes = (*NumericTensorTypes, Variable, ndarray)
ArrayType = Union[ArrayTypes]

str_to_int_tensor_type = {'int': get_default_int_size(),
                          'int8': CharTensor, 'uint8': ByteTensor, 'int16': ShortTensor,
                          'int32': IntTensor, 'int64': LongTensor,
                          'short': ShortTensor, 'byte': ByteTensor, 'char': CharTensor, 'long': LongTensor}


def Identity(X: T1) -> T1:
    return X


#####################################################################
# Sequence data utils                                               #
#####################################################################

class Vocabulary:
    """
    A two-way hash table mapping entities to int IDs and a reverse hash mapping IDs to entities.
    """
    def __init__(self):
        # hash unigram --> ID
        self.ID = dict()
        # hash ID --> unigram
        self.token = dict()

        self.size = 0
        self.docCount = 0
        self.maxID = -1

    def add(self, token: Hashable):
        self.addMany((token,))

    def addMany(self, tokens: Iterable[Hashable]):
        for token in tokens:
            if token not in self.ID:
                # increment the maxID and vocabSize
                self.maxID += 1
                self.size += 1
                # set both mappings
                self.ID[token] = self.maxID
                self.token[self.maxID] = token


#####################################################################
# Data set utils                                                    #
#####################################################################

def train_valid_test_split(data: Union[Seq[T1], int], *args: Seq[float]):
    def get_idxs(ixs):
        return [data[i] for i in ixs]

    if isinstance(data, int):
        idxs = list(range(data))
    else:
        idxs = list(range(len(data)))

    if len(args) == 3:
        trainp, validp, testp = args
    elif len(args) == 2:
        trainp, validp = args
        testp = 1.0 - trainp - validp
    elif len(args) == 1:
        trainp = args[0]
        testp = 1.0 - trainp
        validp = None
    else:
        raise ValueError("*args must be length 1, 2, or 3")

    tot = 0.0
    for x in (trainp, validp, testp):
        if x is not None:
            assert 0.0 < x < 1.0
            tot += x
    assert np_abs(1.0 - tot) <= finfo(float).resolution

    trainn = round(trainp * len(idxs))
    train_idx = sample(idxs, trainn)

    if validp is None:
        test_idx = list(set(idxs).difference(train_idx))
        return_idxs = train_idx, test_idx
    else:
        validn = round(validp * len(idxs))
        testn = len(idxs) - trainn - validn
        nontrain_idx = set(idxs).difference(train_idx)
        valid_idx = sample(nontrain_idx, validn)
        test_idx = list(set(nontrain_idx).difference(valid_idx))
        return_idxs = train_idx, valid_idx, test_idx

    return tuple(map(get_idxs, return_idxs))


def to_tensor(arr: ArrayType):
    if isinstance(arr, _TensorBase):
        return arr
    elif isinstance(arr, Variable):
        return arr.data
    elif isinstance(arr, ndarray):
        return from_numpy(arr)
    else:
        raise TypeError("`arr` must be an instance of one of {}".format(ArrayTypes))


def efficient_batch_iterator(X: Iterable[T1], y: Opt[Iterable[T2]]=None,
                             X_encoder: Opt[Callable[[T1], TensorType]]=None,
                             y_encoder: Opt[Callable[[T2], TensorType]]=None,
                             batch_size: int=32, shuffle: bool=False,
                             num_workers: int=0, classifier: bool=False) -> Iterable[Tuple[TensorType, TensorType]]:
    num_workers = get_torch_num_workers(num_workers)
    if y is None:
        # for, e.g. autoencoders
        y = X
    if isinstance(X, ArrayTypes):
        if isinstance(y, ArrayTypes):
            # encoders should take batch tensors in this instance
            dataset = TensorDataset(to_tensor(X), to_tensor(y))
            return MappedDataLoader(X_encoder=X_encoder, y_encoder=y_encoder, dataset=dataset, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=num_workers,
                                    classifier=classifier)
    elif isinstance(X, Seq) or (hasattr(X, '__len__') and hasattr(X, '__getitem__')):
        if isinstance(y, Seq) or (hasattr(y, '__len__') and hasattr(y, '__getitem__')):
            # Seq has __len__ and __getitem__ so it can serve as a dataset
            dataset = SeqDataset(X, y, X_encoder=X_encoder, y_encoder=y_encoder)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers)
    elif isinstance(X, Iterable):
        if isinstance(y, Iterable):
            return BatchedIterableDataloader(X, y, batch_size=batch_size, X_encoder=X_encoder, y_encoder=y_encoder,
                                             classifier=classifier, num_workers=num_workers)
    else:
        raise TypeError("`X` and `y` must both be array types, numeric sequences, or iterables.")


class SeqDataset(Dataset):
    """for feeding to a torch.utils.data.DataLoader"""
    def __init__(self, X_seq: Seq[T1], y_seq: Seq[T2],
                 X_encoder: Opt[Callable[[T1], TensorType]]=None, y_encoder: Opt[Callable[[T2], Numeric]]=None):
        assert len(X_seq) == len(y_seq)
        self.X_seq = X_seq
        self.y_seq = y_seq
        self.X_encoder = X_encoder if X_encoder is not None else Identity
        self.y_encoder = y_encoder if y_encoder is not None else Identity

    def __len__(self) -> int:
        return len(self.X_seq)

    def __getitem__(self, item) -> TensorType:
        return self.X_encoder(self.X_seq[item]), self.y_encoder(self.y_seq[item])


class BatchedIterableDataloader:
    """as a stand-in for a torch.utils.data.DataLoader when random access is unavailable or length is unknown"""
    def __init__(self, X_iterable: Iterable[T1], y_iterable: Iterable[T2], batch_size: int,
                 X_encoder: Opt[Callable[[T1], TensorType]]=None, y_encoder: Opt[Callable[[T2], Numeric]]=None,
                 classifier: bool=False, num_workers: Opt[int]=None):
        assert isinstance(y_iterable, Iterable)
        assert isinstance(X_iterable, Iterable)
        if num_workers and num_workers != 0:
            print("Warning: {} does not support multithreaded data loading; "
                  "loading data in the main thread instead.".format(self.__class__.__name__))

        self.X_iterable = X_iterable
        self.y_iterable = y_iterable
        self.X_encoder = X_encoder
        self.y_encoder = y_encoder
        self.batch_size = batch_size
        self.stack_y = stack if not classifier else LongTensor

    def __iter__(self) -> Iterable[Tuple[TensorType, TensorType]]:
        X_ = map(self.X_encoder, self.X_iterable) if self.X_encoder is not None else self.X_iterable
        y_ = map(self.y_encoder, self.y_iterable) if self.y_encoder is not None else self.y_iterable
        b = self.batch_size
        batches = zip(batched(X_, batch_size=b), batched(y_, batch_size=b))
        stack_y = self.stack_y
        return ((stack(xs), stack_y(ys)) for xs, ys in batches)


class MappedDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, X_encoder: Opt[Callable[[T1], TensorType]]=None,
                 y_encoder: Opt[Callable[[T2], TensorType]]=None, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.X_encoder = X_encoder
        self.y_encoder = y_encoder

    def __iter__(self) -> Iterable[Tuple[TensorType, TensorType]]:
        iterator = super().__iter__()
        if self.X_encoder is not None and self.y_encoder is not None:
            return ((self.X_encoder(X), self.y_encoder(y)) for X, y in iterator)
        elif self.X_encoder is not None:
            return ((self.X_encoder(X), y) for X, y in iterator)
        elif self.y_encoder is not None:
            return ((X, self.y_encoder(y)) for X, y in iterator)
        else:
            return iterator


class TupleIteratorDataLoader:
    def __init__(self, dataset: Iterable[Tuple[T1, T2]], batch_size: int,
                 X_encoder: Opt[Callable[[T1], TensorType]]=None,
                 y_encoder: Opt[Callable[[T2], TensorType]]=None, classifier=False,
                 num_workers: Opt[int]=None):
        assert isinstance(dataset, Iterable)
        if num_workers and num_workers != 0:
            print("Warning: {} does not support multithreaded data loading; "
                  "loading data in the main thread instead.".format(self.__class__.__name__))

        self.dataset = dataset
        self.X_encoder = X_encoder
        self.y_encoder = y_encoder
        self.batch_size = batch_size
        self.stack_y = stack if not classifier else LongTensor

    def __iter__(self) -> Iterable[Tuple[TensorType, TensorType]]:
        iterator = iter(self.dataset)
        if self.X_encoder is not None and self.y_encoder is not None:
            instances = ((self.X_encoder(x), self.y_encoder(y)) for x, y in iterator)
        elif self.X_encoder is not None:
            instances = ((self.X_encoder(X), y) for X, y in iterator)
        elif self.y_encoder is not None:
            instances = ((X, self.y_encoder(y)) for X, y in iterator)
        else:
            instances = iterator

        batches = (tuple(zip(*batch)) for batch in batched(instances, batch_size=self.batch_size))
        stack_y = self.stack_y
        return ((stack(xs), stack_y(ys)) for xs, ys in batches)


class RNNSequencePredictorDataset(Dataset):
    """Subclass of torch.utils.data.Dataset - can be safely passed to a torch.utils.data.DataLoader for multithreaded
    data loading. Assumes a sequence prediction (rather than classification) task; dependent variable is simply the
    independent variable sequence offset by 1."""
    def __init__(self, sequences: Iterable[Seq[T1]], seq_len: int, vocab: Opt[Vocabulary]=None,
                 build_vocab: bool=False, null_token: T1=None, int_id_type: str='long', shuffle:bool=True):
        """
        :param sequences:
        :param seq_len: length of sequences to return in training batches
        :param vocab: Optional instance of data.
        :param build_vocab: if no vocab is passed, should one be built for encoding? Can be False if
        :param null_token: Optional hashable to use for padding sequences.
        :param int_id_type: string indicating the type of int ids to use. Must be a key of data.str_to_int_tensor_type.
            Numpy aliases for integer types are valid, as well as 'long', 'short', 'byte', 'char'
        :param shuffle: bool; whether to shuffle examples when iterating over this dataset directly (as opposed to
            using a DataLoader to load batches)
        """
        self.vocab = vocab
        self.null_token = null_token
        tensor = str_to_int_tensor_type[int_id_type]
        self.tensor_type = tensor
        self.shuffle = shuffle

        if vocab is None and build_vocab:
            vocab = Vocabulary()
            vocab.addMany(chain.from_iterable(sequences))
            self.vocab = vocab
        elif vocab is None:
            self.encode = Identity
        else:
            self.vocab.add(null_token)

        self.null_id = self.vocab.ID[self.null_token]
        self.seq_len = seq_len

        def ix_pairs(tup):
            i, seq = tup
            return zip(repeat(i), range(max(seq_len - len(seq) + 1, 1)))

        self.sequences = [tensor(self.pad(self.encode(s))) for s in sequences]
        self.seq_idxs = list(chain.from_iterable(map(ix_pairs, enumerate(self.sequences))))

    def encode(self, tokens: Seq[T1]) -> List[int]:
        encoder = self.vocab.ID.__getitem__
        return list(map(encoder, tokens))

    def pad(self, tokens: List[int]) -> List[int]:
        return list(chain(tokens, repeat(self.null_id, (self.seq_len + 1 - len(tokens)))))

    def __len__(self):
        return len(self.seq_idxs)

    def __getitem__(self, idx) -> IntTensorType:
        seq_idx, position = self.seq_idxs[idx]
        seq = self.sequences[seq_idx]
        x = seq[position:position + self.seq_len]
        y = seq[(position + 1):(position + self.seq_len + 1)]
        return x, y

    def __iter__(self):
        idxs = range(len(self))
        if self.shuffle:
            idxs = sample(list(idxs), len(self))
        getsample = self.__getitem__
        return (getsample(i) for i in idxs)
