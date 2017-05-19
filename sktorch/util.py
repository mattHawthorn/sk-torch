#coding:utf-8
from typing import Union, Tuple, Iterable, Iterator, Callable, TypeVar, Sequence as Seq, Optional as Opt
from numbers import Number
from os import cpu_count
from itertools import islice, chain

from numpy import ndarray
from torch import from_numpy
from torch import stack
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import cuda, backends, HalfTensor, FloatTensor, DoubleTensor, IntTensor, LongTensor, _TensorBase

T1 = TypeVar('T1')
T2 = TypeVar('T2')
NumericTensorTypes = (FloatTensor, HalfTensor, DoubleTensor, IntTensor, LongTensor)
Tensor = Union[NumericTensorTypes]
Numeric = Union[Number, Tensor, Variable, ndarray]
ArrayTypes = (*NumericTensorTypes, Variable, ndarray)


def cuda_available():
    return cuda.is_available() and backends.cudnn.enabled


def get_torch_num_workers(num_workers: int):
    """turn an int into a useful number of workers for a pytorch DataLoader.
    -1 means "use all CPU's", -2, means "use all but 1 CPU", etc.
    Note: 0 is interpreted by pytorch as doing data loading in the main process, while any positive number spawns a
    new process. We do not allow more processes to spawn than there are CPU's."""
    num_cpu = cpu_count()
    if num_workers < 0:
        n_workers = num_cpu + 1 + num_workers
        if n_workers < 0:
            print("Warning: {} fewer workers than the number of CPU's were specified, but there are only {} CPU's; "
                  "running data loading in the main process (num_workers = 0).".format(num_workers + 1, num_cpu))
        num_workers = max(0, n_workers)
    if num_workers > num_cpu:
        print("Warning, `num_workers` is {} but only {} CPU's are available; "
              "using this number instead".format(num_workers, num_cpu))
    return min(num_workers, num_cpu)


def peek(iterable: Iterable, n: int):
    """safe peek of head of iterable/iterator without consuming"""
    if isinstance(iterable, Iterator):
        peek_ = list(islice(iterable, n))
        return peek_, chain(peek_, iterable)
    elif isinstance(iterable, Iterable):
        peek_ = list(islice(iter(iterable), n))
        return peek_, iterable
    else:
        raise TypeError("`iterable` must be an iterable")


def batched(items, batch_size=None):
    items = iter(items)
    if batch_size is None:
        yield items
    else:
        while True:
            iterslice = list(islice(items, batch_size))
            if len(iterslice) == 0:
                break
            yield iterslice


def to_tensor(arr: Union[ArrayTypes]):
    if isinstance(arr, _TensorBase):
        return arr
    elif isinstance(arr, Variable):
        return arr.data
    elif isinstance(arr, ndarray):
        return from_numpy(arr)
    else:
        raise TypeError("`arr` must be an instance of one of {}".format(ArrayTypes))


def efficient_batch_iterator(X: Iterable[T1], y: Opt[Iterable[T2]]=None,
                             X_encoder: Opt[Callable[[T1], Tensor]]=None,
                             y_encoder: Opt[Callable[[T2], Tensor]]=None,
                             batch_size: int=32, shuffle: bool=False,
                             num_workers: int=0, classifier: bool=False) -> Iterable[Tuple[Tensor, Tensor]]:
    if y is None:
        # for, e.g. autoencoders
        y = X
    if isinstance(X, ArrayTypes):
        if isinstance(y, ArrayTypes):
            # encoders should take batch tensors in this instance
            dataset = TensorDataset(to_tensor(X), to_tensor(y))
            return MappedDataLoader(X_encoder=X_encoder, y_encoder=y_encoder, dataset=dataset, batch_size=batch_size, 
                                    shuffle=shuffle, num_workers=get_torch_num_workers(num_workers),
                                    classifier=classifier)
    elif isinstance(X, Seq) or (hasattr(X, '__len__') and hasattr(X, '__getitem__')):
        if isinstance(y, Seq) or (hasattr(y, '__len__') and hasattr(y, '__getitem__')):
            # Seq has __len__ and __getitem__ so it can serve as a dataset
            dataset = SeqDataset(X, y, X_encoder=X_encoder, y_encoder=y_encoder)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=get_torch_num_workers(num_workers))
    elif isinstance(X, Iterable):
        if isinstance(y, Iterable):
            return BatchedIterableDataloader(X, y, batch_size=batch_size, X_encoder=X_encoder, y_encoder=y_encoder,
                                             classifier=classifier)
    else:
        raise TypeError("`X` and `y` must both be array types, numeric sequences, or iterables.")


def Identity(X: T1) -> T1:
    return X


class SeqDataset(Dataset):
    """for feeding to a torch.utils.data.DataLoader"""
    def __init__(self, X_seq: Seq[T1], y_seq: Seq[T2],
                 X_encoder: Opt[Callable[[T1], Tensor]]=None, y_encoder: Opt[Callable[[T2], Numeric]]=None):
        assert len(X_seq) == len(y_seq)
        self.X_seq = X_seq
        self.y_seq = y_seq
        self.X_encoder = X_encoder if X_encoder is not None else Identity
        self.y_encoder = y_encoder if y_encoder is not None else Identity

    def __len__(self) -> int:
        return len(self.X_seq)

    def __getitem__(self, item) -> Tensor:
        return self.X_encoder(self.X_seq[item]), self.y_encoder(self.y_seq[item])


class BatchedIterableDataloader:
    """as a stand-in for a torch.utils.data.DataLoader when random access is unavailable or length is unknown"""
    def __init__(self, X_iterable: Iterable[T1], y_iterable: Iterable[T2], batch_size: int,
                 X_encoder: Opt[Callable[[T1], Tensor]]=None, y_encoder: Opt[Callable[[T2], Numeric]]=None,
                 classifier: bool=False):
        assert isinstance(X_iterable, Iterable)
        assert isinstance(y_iterable, Iterable)
        self.X_iterable = X_iterable
        self.y_iterable = y_iterable
        self.X_encoder = X_encoder
        self.y_encoder = y_encoder
        self.batch_size = batch_size
        self.stack_y = stack if not classifier else LongTensor

    def __iter__(self) -> Iterable[Tuple[Tensor, Tensor]]:
        X_ = map(self.X_encoder, self.X_iterable) if self.X_encoder is not None else self.X_iterable
        y_ = map(self.y_encoder, self.y_iterable) if self.y_encoder is not None else self.y_iterable
        b = self.batch_size
        batches = zip(batched(X_, batch_size=b), batched(y_, batch_size=b))
        stack_y = self.stack_y
        return ((stack(xs), stack_y(ys)) for xs, ys in batches)


class MappedDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, X_encoder: Opt[Callable[[T1], Tensor]]=None,
                 y_encoder: Opt[Callable[[T2], Tensor]]=None, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.X_encoder = X_encoder
        self.y_encoder = y_encoder

    def __iter__(self) -> Iterable[Tuple[Tensor, Tensor]]:
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
                 X_encoder: Opt[Callable[[T1], Tensor]]=None,
                 y_encoder: Opt[Callable[[T2], Tensor]]=None, classifier=False):
        self.dataset = dataset
        self.X_encoder = X_encoder
        self.y_encoder = y_encoder
        self.batch_size = batch_size
        self.stack_y = stack if not classifier else LongTensor

    def __iter__(self) -> Iterable[Tuple[Tensor, Tensor]]:
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
