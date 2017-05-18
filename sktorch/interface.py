#coding:utf-8

from typing import Any, Iterable, Callable, Union, Sequence as Seq, Optional as Opt
from collections import deque
from numpy import ndarray
import torch
from torch import autograd
from torch import nn
from torch import optim
from torch.nn.modules import loss
from .util import cuda_available, peek, efficient_batch_iterator, T1, T2, Tensor

DEFAULT_BATCH_SIZE = 32


def training_mode(mode: bool):
    """decorator factory to make a decorator to set the training mode of an NN with a pytorch backend"""
    def dec(nn_method):
        def method(obj: 'TorchModel', *args, **kwargs):
            obj.set_mode(mode)
            result = nn_method(obj, *args, **kwargs)
            obj.set_mode(False)
            return result
        return method
    return dec


class TorchModel:
    """Wrapper class to handle encoding inputs to pytorch variables, managing transfer to/from the GPU,
    handling train/eval mode, etc."""
    def __init__(self, torch_module: nn.Module, loss_func: loss._Loss,
                 optimizer: optim.Optimizer,
                 optimizer_kwargs: Opt[dict]=None,
                 input_encoder: Opt[Callable[[T1], Tensor]]=None,
                 output_encoder: Opt[Callable[[T2], Tensor]]=None,
                 output_decoder: Opt[Callable[[Tensor], T2]] = None,
                 estimate_normalization_samples: Opt[int]=None,
                 print_func: Callable[[Any], None]=print, num_dataloader_workers: int=-2):
        """
        :param torch_module: a torch.nn.Module
        :param loss_func: a torch.nn.modules.loss._Loss callable
        :param optimizer: a torch.optim.Optimizer
        :param input_encoder: a callable taking the type of the training data and encoding it to tensors for the
            forward pass in the torch module
        :param estimate_normalization_samples: If normalization of inputs is called for, use this many samples of
            training data to estimate the mean and sd per input dimension
        :param print_func: callable with no return value, ideally prints to screen or log file
        """
        self.gpu_enabled = cuda_available()
        self._torch_module = None
        self._optimizer = None

        # property setter method ensures this goes to the gpu if it's available
        self.torch_module = torch_module

        # property setter method gets the torch.optim class if this is a string, checks inheritance, passes
        # module params and optimizer_kwargs to constructor
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = optimizer

        self.loss_func = loss_func
        # you could pass in a logger.info/debug or a file.write method for this if you like
        self.print = print_func

        self.should_normalize = (estimate_normalization_samples is not None)
        self.norm_n_samples = estimate_normalization_samples
        self._input_mean = None
        self._input_sd = None
        self._norm_estimated = False

        self.encode_input = input_encoder
        self.encode_target = output_encoder
        if output_decoder is not None:
            self.decode_output = output_decoder

        # these take tensors and wrap them in Variables and move them to the GPU if necessary
        self.prepare_input = self.get_input_preparer()
        self.prepare_target = self.get_target_preparer()

        self.num_dataloader_workers = num_dataloader_workers

    def estimate_normalization(self, sample: Union[torch.FloatTensor, ndarray]):
        sample = sample[0:self.norm_n_samples]
        mean = sample.mean(0)
        sd = sample.std(0)
        self._input_mean = mean.cuda() if self.gpu_enabled else mean.cpu()
        self._input_sd = sd.cuda() if self.gpu_enabled else sd.cpu()
        self._norm_estimated = True

    def normalize(self, X: Union[torch.FloatTensor, autograd.Variable]):
        if not self._norm_estimated:
            raise ValueError("normalization constants have not yet been estimated")
        normed = (X - self._input_mean.expand_as(X))
        # can do this operation in place
        normed /= self._input_sd.expand_as(X)
        return normed

    @property
    def input_mean(self):
        # no setting allowed for this - don't want to mess it up!
        return self._input_mean

    @property
    def input_sd(self):
        # no setting allowed for this - don't want to mess it up!
        return self._input_sd

    def get_input_preparer(self) -> Callable[[torch.FloatTensor], autograd.Variable]:
        if self.should_normalize:
            if self.gpu_enabled:
                def prepare(data: torch.FloatTensor) -> autograd.Variable:
                    return autograd.Variable(self.normalize(data.cuda()), volatile=not self._torch_module.training)
            else:
                def prepare(data: torch.FloatTensor) -> autograd.Variable:
                    return autograd.Variable(self.normalize(data.cpu()), volatile=not self._torch_module.training)
        else:
            if self.gpu_enabled:
                def prepare(data: torch.FloatTensor) -> autograd.Variable:
                    return autograd.Variable(data.cuda(), volatile=not self._torch_module.training)
            else:
                def prepare(data: torch.FloatTensor) -> autograd.Variable:
                    return autograd.Variable(data.cpu(), volatile=not self._torch_module.training)
        return prepare

    def get_target_preparer(self) -> Callable[[torch.FloatTensor], autograd.Variable]:
        if self.gpu_enabled:
            def prepare(data: torch.FloatTensor) -> autograd.Variable:
                return autograd.Variable(data.cuda(), requires_grad=False, volatile=not self._torch_module.training)
        else:
            def prepare(data: torch.FloatTensor) -> autograd.Variable:
                return autograd.Variable(data.cpu(), requires_grad=False, volatile=not self._torch_module.training)
        return prepare

    @property
    def torch_module(self):
        return self._torch_module

    @torch_module.setter
    def torch_module(self, module: nn.Module):
        self._torch_module = module.cuda() if self.gpu_enabled else module.cpu()

    @property
    def parameters(self):
        return list(self.torch_module.parameters())

    @property
    def optimizer(self):
        if self.optimizer_kwargs:
            return self._optimizer(self.torch_module.parameters(), **self.optimizer_kwargs)
        else:
            return self._optimizer(self.torch_module.parameters())

    @optimizer.setter
    def optimizer(self, optimizer: Union[str, type]):
        if isinstance(optimizer, str):
            optimizer = getattr(optim, optimizer)
        if not issubclass(optimizer, optim.Optimizer):
            raise TypeError("`optimizer` must be a torch.optim.Optim or a string which refers to one by name")
        self._optimizer = optimizer

    def set_mode(self, training: bool):
        if self.torch_module.training != training:
            self.torch_module.train(training)

    def _single_batch_train_pass(self, X_batch: Tensor, y_batch: Tensor, optimizer: optim.Optimizer):
        module = self.torch_module
        module.zero_grad()
        optimizer.zero_grad()
        err = self._single_batch_test_pass(X_batch, y_batch)
        err.backward()
        optimizer.step()
        return err

    def _single_batch_test_pass(self, X_batch: Tensor, y_batch: Tensor):
        y_batch = self.prepare_target(y_batch)
        output = self._single_batch_forward_pass(X_batch)
        err = self.loss_func(output, y_batch)
        return err

    def _single_batch_forward_pass(self, X_batch: Tensor):
        X_batch = self.prepare_input(X_batch)
        output = self.torch_module(X_batch)
        return output

    @training_mode(True)
    def fit(self, X: Iterable[T1], y: Iterable[T2], X_test: Opt[Iterable[T1]]=None, y_test: Opt[Iterable[T1]]=None,
            batch_size: int=DEFAULT_BATCH_SIZE, shuffle: bool=False, epochs: int=1,
            min_epochs: int=1, min_rel_improvement: float=1e-5, epochs_without_improvement: int=5,
            batch_report_interval: Opt[int]=None, epoch_report_interval: Opt[int]=None):
        if self.should_normalize:
            sample, X = peek(X, self.norm_n_samples)
            if self.encode_input:
                sample = torch.stack([self.encode_input(x) for x in sample])
            self.estimate_normalization(sample)

        self.update(X=X, y=y, X_test=X_test, y_test=y_test,
                    batch_size=batch_size, shuffle=shuffle,
                    epochs=epochs, min_epochs=min_epochs,
                    min_rel_improvement=min_rel_improvement,
                    epochs_without_improvement=epochs_without_improvement,
                    batch_report_interval=batch_report_interval,
                    epoch_report_interval=epoch_report_interval)

    @training_mode(True)
    def update(self, X: Iterable[T1], y: Iterable[T1], X_test: Opt[Iterable[T1]]=None, y_test: Opt[Iterable[T1]]=None,
               batch_size: int=DEFAULT_BATCH_SIZE, shuffle: bool=False, epochs: int=1,
               min_epochs: int=1, min_rel_improvement: Opt[float]=1e-5, epochs_without_improvement: int=5,
               batch_report_interval: Opt[int]=None, epoch_report_interval: Opt[int]=None):
        assert epochs > 0

        dataset = efficient_batch_iterator(X, y, X_encoder=self.encode_input, y_encoder=self.encode_target,
                                           batch_size=batch_size, shuffle=shuffle,
                                           num_workers=self.num_dataloader_workers)
        if X_test is not None and y_test is not None:
            test_data = efficient_batch_iterator(X_test, y_test, X_encoder=self.encode_input,
                                                 y_encoder=self.encode_target,
                                                 batch_size=batch_size, shuffle=shuffle,
                                                 num_workers=self.num_dataloader_workers)
        else:
            if X_test is not None or y_test is not None:
                self.print("Warning: test data was provided but either the regressors or the response were omitted")
            test_data = None

        if test_data is None:
            losstype = 'training'
        else:
            losstype = 'test'

        optimizer = self.optimizer

        epoch, epoch_loss = 0, 0.0
        epoch_losses = deque(maxlen=epochs_without_improvement)

        for epoch in range(epochs):
            if epoch_report_interval and epoch % epoch_report_interval == epoch_report_interval - 1:
                self.print("Training epoch {}".format(epoch + 1))

            running_loss = 0.0
            running_samples = 0
            for i, (X_batch, y_batch) in enumerate(dataset):
                n_samples = X_batch.size()[0]
                running_samples += n_samples

                err = self._single_batch_train_pass(X_batch, y_batch, optimizer)
                running_loss += err.data[0]
                err = err / n_samples

                if batch_report_interval and i % batch_report_interval == batch_report_interval - 1:
                    self.report_batch(epoch, i, err.data[0])

            epoch_loss = running_loss / running_samples
            if test_data is not None:
                epoch_loss = self._error(test_data)

            if epoch_report_interval and epoch % epoch_report_interval == epoch_report_interval - 1:
                self.report_epoch(epoch, epoch_loss, losstype)

            if epoch + 1 >= min_epochs and self.stop_training(epoch_loss, epoch_losses, min_rel_improvement):
                break
            # print(epoch_loss, epoch_losses, min_rel_improvement)
            # print(self.stop_training(epoch_loss, epoch_losses, min_rel_improvement))

            epoch_losses.append(epoch_loss)

        if epoch_report_interval:  # and epoch % epoch_report_interval != epoch_report_interval - 1:
            self.report_epoch(epoch, epoch_loss, losstype)

    def report_epoch(self, epoch: int, epoch_loss: float, losstype: str):
        lossname = self.loss_func.__class__.__name__
        self.print("epoch {}, {} {} per sample: {}".format(epoch + 1, losstype, lossname, epoch_loss))

    def report_batch(self, epoch: int, batch: int, batch_loss: float):
        lossname = self.loss_func.__class__.__name__
        self.print("epoch {}, batch {}, {} per sample: {}".format(epoch + 1, batch + 1, lossname, batch_loss))

    def stop_training(self, epoch_loss: float, epoch_losses: Seq[float], min_rel_improvement: float):
        improvements = [(l - epoch_loss) / l for l in epoch_losses]
        stop = len(epoch_losses) > 0 and min_rel_improvement is not None and \
            all(i < min_rel_improvement for i in improvements)
        if stop:
            self.print("No relative loss improvement greater than {}% over last {} epochs; stopping".format(
                round(min_rel_improvement * 100.0, 4), len(epoch_losses)
            ))
        return stop

    @training_mode(False)
    def error(self, X, y, batch_size: int = DEFAULT_BATCH_SIZE, shuffle: bool=False):
        dataset = efficient_batch_iterator(X, y, X_encoder=self.encode_input, y_encoder=self.encode_target,
                                           batch_size=batch_size, shuffle=shuffle, 
                                           num_workers=self.num_dataloader_workers)
        return self._error(dataset)

    def _error(self, dataset):
        running_loss = 0.0
        running_samples = 0
        for X_batch, y_batch in dataset:
            err = self._single_batch_test_pass(X_batch, y_batch)
            running_loss += err.data[0]
            running_samples += X_batch.size()[0]
        return running_loss / running_samples

    @training_mode(False)
    def predict(self, X: Iterable[Any], batch_size: int = DEFAULT_BATCH_SIZE, shuffle: bool=False) -> Iterable[Any]:
        dataset = efficient_batch_iterator(X, X_encoder=self.encode_input, y_encoder=self.encode_target,
                                           batch_size=batch_size, shuffle=shuffle, 
                                           num_workers=self.num_dataloader_workers)
        for X_batch, X_batch2 in dataset:
            for output in self._single_batch_forward_pass(X_batch):
                yield self.decode_output(output)
    
    @staticmethod
    def encode_input(X: T1) -> Tensor:
        """encode the input to a tensor that can be fed to the neural net;
        this can be passed to the class constructor for customizability, else it is assumed to be the identity."""
        return X

    @staticmethod
    def encode_output(y: T2) -> Tensor:
        """encode the output to a tensor that can be used to compute the error of a neural net prediction;
        this can be passed to the class constructor for customizability, else it is assumed to be the identity."""
        return y

    @staticmethod
    def decode_output(y: Iterable[Tensor]) -> T2:
        """take the output Variable from the neural net and decode it to whatever type the training set target was;
        this can be passed to the class constructor for customizability, else it is assumed to be the identity."""
        return y
