#coding:utf-8
from typing import Tuple, Iterable, Iterator, IO, Union, Optional as Opt
from os import cpu_count
from io import BytesIO
from tempfile import TemporaryFile
from platform import architecture
from itertools import islice, chain

from numpy import log10, floor
from torch import cuda, backends, IntTensor, LongTensor, save as torch_save, load as torch_load


#####################################################################
# IO utils                                                          #
#####################################################################

time_units = {0: 's', 3: 'ms', 6: '\u03BCs', 9: 'ns'}
def pretty_time(t: float):
    t = float(t)
    move_right = -int(floor(log10(t)))
    unit = 0 if move_right <= 0 else 3*(move_right//3) + (3 if move_right % 3 > 0 else 0)
    unit = min(unit, 9)
    return str(round(t*10**unit, 4)) + time_units[unit]


def open_file(path: Union[str, IO], mode='rb'):
    if isinstance(path, str):
        file = open(path, mode)
    else:
        file = path
    return file


def get_torch_object_bytes(obj):
    with TemporaryFile() as f:
        torch_save(obj, f)
        f.seek(0)
        b = f.read()
    return b


def load_torch_object_bytes(b):
    with TemporaryFile() as f:
        f.write(b)
        f.seek(0)
        obj = torch_load(f)
    return obj


#####################################################################
# Hardware utils                                                    #
#####################################################################

def cuda_available():
    return cuda.is_available() and backends.cudnn.enabled


def get_default_int_size():
    arch = architecture()
    bitstr = arch[0]
    bits = 32 if '32' in bitstr else (64 if '64' in bitstr else None)
    tensor_type = LongTensor if bits == 64 else (IntTensor if bits == 32 else None)
    return tensor_type


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


#####################################################################
# Stopping criteria                                                 #
#####################################################################

def last_epoch_min_rel_improvement(epoch_losses: Iterable[float], min_rel_improvement: float) -> Tuple[bool, Opt[str]]:
    """return value indicates whether the last epoch's loss is at least min_rel_improvement better than all prior
    epochs, as a proportion of each prior epoch loss."""
    epoch_losses = list(epoch_losses)
    prior_losses = epoch_losses[:-1]
    last_loss = epoch_losses[-1]
    improvements = [(l - last_loss) / l for l in prior_losses]
    stop = len(prior_losses) > 0 and min_rel_improvement is not None and \
        all(i < min_rel_improvement for i in improvements)
    message = "No relative loss improvement greater than {}% over last {} epochs; stopping".format(
                round(min_rel_improvement * 100.0, 4), len(epoch_losses)) if stop else None
    return stop, message


#####################################################################
# Iterator utils                                                    #
#####################################################################

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
