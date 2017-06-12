#coding:utf-8
from typing import Tuple, Iterable, Optional as Opt
from functools import reduce
from itertools import starmap
from operator import and_, gt
from numpy import array


#####################################################################
# Stopping criteria                                                 #
#####################################################################

class tail_losses_no_relative_improvement:
    """Relative improvement of the last loss over each of the prior tail_len - 1 losses does not exceed
    min_rel_improvement, as a proportion of the prior losses."""

    def __init__(self, tail_len: int=4, min_rel_improvement: float=1e-4):
        self.tail_len = tail_len
        self.min_rel_improvement = min_rel_improvement

    def __call__(self, epoch_losses: Iterable[float], test_losses: Iterable[float]) -> Tuple[bool, Opt[str]]:
        tail_len, min_rel_improvement = self.tail_len, self.min_rel_improvement
        test_losses = list(test_losses) if not isinstance(test_losses, list) else test_losses
        if len(test_losses) < 2:
            return False, None
        prior_losses = test_losses[-min(len(test_losses), tail_len):-1]
        last_loss = test_losses[-1]
        improvements = [(l - last_loss) / l if l != 0.0 else 0.0 for l in prior_losses]
        stop = min_rel_improvement is not None and all(i < min_rel_improvement for i in improvements)
        message = "No relative loss improvement greater than {}% over last {} epochs; stopping".format(
                    round(min_rel_improvement * 100.0, 4), len(prior_losses) + 1) if stop else None
        return stop, message


class tail_losses_n_consecutive_increases:
    """Validation loss has increased for n_consecutive_increases epochs.
    Does not generally work for training loss."""
    def __init__(self, n_consecutive_increases: int=2):
        self.n_consecutive_increases = n_consecutive_increases

    def __call__(self, epoch_losses: Iterable[float], test_losses: Iterable[float]) -> Tuple[bool, Opt[str]]:
        n_consecutive_increases = self.n_consecutive_increases
        test_losses = list(test_losses) if not isinstance(test_losses, list) else test_losses
        if len(test_losses) < n_consecutive_increases + 1:
            return False, None
        tail = test_losses[-(n_consecutive_increases + 1):]
        stop = reduce(and_, starmap(gt, zip(tail[1:], tail)))
        message = "{} consecutive loss increases over last {} epochs; stopping".format(
                    n_consecutive_increases, n_consecutive_increases) if stop else None
        return stop, message


class max_generalization_loss:
    """Current (validation loss / optimal validation loss so far) - 1 >= generalization_loss.
    Does not generally work for training loss."""
    def __init__(self, generalization_loss: float=0.05):
        self.generalization_loss = generalization_loss

    def __call__(self, epoch_losses: Iterable[float], test_losses: Iterable[float]) -> Tuple[bool, Opt[str]]:
        generalization_loss = self.generalization_loss
        test_losses = array(list(test_losses)) if not isinstance(test_losses, list) else array(test_losses)
        if len(test_losses) < 3:
            return False, None
        min_loss = test_losses.min()
        gen_loss = test_losses[-1] / min_loss - 1.0
        stop = gen_loss >= generalization_loss
        message = "Relative loss increase of {}% over minimum so far; stopping".format(
                    round(gen_loss * 100.0, 4)) if stop else None
        return stop, message
