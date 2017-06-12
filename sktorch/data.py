#coding:utf-8
from numbers import Number
from random import sample
from typing import Sequence as Seq, Optional as Opt
from typing import Union, Tuple, Iterable, Hashable, Callable, TypeVar

from numpy import ndarray, finfo, abs as np_abs
from torch import HalfTensor, FloatTensor, DoubleTensor, ByteTensor, CharTensor, ShortTensor, IntTensor, LongTensor, _TensorBase
from torch import from_numpy
from torch import stack
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataloader import default_collate

from .util import batched, get_torch_num_workers, get_default_int_size

#####################################################################
# Type aliases                                                      #
#####################################################################

T1 = TypeVar('T1')
T2 = TypeVar('T2')
H = TypeVar('H', bound=Hashable)

FloatTensorTypes = (FloatTensor, HalfTensor, DoubleTensor)
FloatTensorType = Union[FloatTensorTypes]

IntTensorTypes = (ByteTensor, CharTensor, ShortTensor, IntTensor, LongTensor)
IntTensorType = Union[IntTensorTypes]

NumericTensorTypes = (*FloatTensorTypes, *IntTensorTypes)
TensorType = Union[NumericTensorTypes]

NumericTypes = (*NumericTensorTypes, Number, Variable, ndarray)
Numeric = Union[NumericTypes]
ArrayTypes = (*NumericTensorTypes, Variable, ndarray)
ArrayType = Union[ArrayTypes]

str_to_int_tensor_type = {'int': get_default_int_size(),
                          'int8': CharTensor, 'uint8': ByteTensor, 'int16': ShortTensor,
                          'int32': IntTensor, 'int64': LongTensor,
                          'short': ShortTensor, 'byte': ByteTensor, 'char': CharTensor, 'long': LongTensor}


def Identity(X: T1) -> T1:
    return X


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
                             num_workers: int=0, classifier: bool=False,
                             collate_fn = default_collate) -> Iterable[Tuple[TensorType, TensorType]]:
    num_workers = get_torch_num_workers(num_workers)
    if y is None:
        # for, e.g. autoencoders
        y = X
    if isinstance(X, ArrayTypes):
        if isinstance(y, ArrayTypes):
            # encoders should take batch tensors in this instance
            dataset = TensorDataset(to_tensor(X), to_tensor(y))
            return MappedDataLoader(X_encoder=X_encoder, y_encoder=y_encoder, dataset=dataset, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn,
                                    classifier=classifier)
    elif isinstance(X, Seq) or (hasattr(X, '__len__') and hasattr(X, '__getitem__')):
        if isinstance(y, Seq) or (hasattr(y, '__len__') and hasattr(y, '__getitem__')):
            # Seq has __len__ and __getitem__ so it can serve as a dataset
            dataset = SeqDataset(X, y, X_encoder=X_encoder, y_encoder=y_encoder)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
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
