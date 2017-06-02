#coding:utf-8

from typing import Any, Tuple, List, Iterable, Callable, Union, IO, Optional as Opt
from collections import deque
import pickle
from time import time
from numpy import ndarray
from torch import autograd, nn, optim, from_numpy, stack
from torch.nn.modules import loss
from torch.nn.utils import clip_grad_norm
from .util import cuda_available, peek, pretty_time, last_epoch_min_rel_improvement
from .util import get_torch_object_bytes, load_torch_object_bytes, open_file
from .data import efficient_batch_iterator, TupleIteratorDataLoader
from .data import T1, T2, TensorType, FloatTensorType, FloatTensorTypes

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
    def __init__(self, torch_module: nn.Module, loss_func: Union[loss._Loss, type, str],
                 optimizer: Union[str, optim.Optimizer],
                 loss_func_kwargs: Opt[dict]=None,
                 optimizer_kwargs: Opt[dict]=None,
                 input_encoder: Opt[Callable[[T1], TensorType]]=None,
                 target_encoder: Opt[Callable[[T2], TensorType]]=None,
                 output_decoder: Opt[Callable[[TensorType], T2]]=None,
                 is_classifier: bool=False,
                 estimate_normalization_samples: Opt[int]=None,
                 print_func: Callable[[Any], None]=print,
                 stopping_criterion: Callable[[List[float], float], Union[bool, Tuple[bool, Opt[str]]]]=
                    last_epoch_min_rel_improvement,
                 num_dataloader_workers: int=-2):
        """
        :param torch_module: a torch.nn.Module
        :param loss_func: a torch.nn.modules.loss._Loss callable
        :param optimizer: a torch.optim.Optimizer
        :param input_encoder: a callable taking the type of the training data independent variable and encoding it to
            tensors for the forward pass in the torch module
        :param target_encoder: a callable taking the type of the training data dependent variable and encoding it to
            tensors for the forward pass in the torch module
        :param output_decoder: a callable taking a (single instance, not batch) torch tensor output of the torch module
            forward pass, and returning the type of the training data dependent variable
        :param estimate_normalization_samples: If normalization of inputs is called for, use this many samples of
            training data to estimate the mean and sd per input dimension
        :param is_classifier: boolean specifying that the target is a single class. This is required to make sure that
            dependent variable batches are collated in the way that torch loss functions expect (1-dimensional)
        :param print_func: callable with no return value, ideally prints to screen or log file
        :param stopping_criterion: callable taking a list of epoch losses and a float criterion and returning either a
            bool or (bool, str or None). The float criterion semantically indicates some 'minimal goodness' or
            'maximal badness' allowed of the sequence of epoch losses. The return bool should indicate whether to
            stop training. The optional return string is a message to be printed at the time that training is stopped.
        :param num_dataloader_workers: int specifying how many threads should be used for data loading. 0 indicates that
            all data loading is done in the main thread (same semantics as torch.utils.data.Dataloader). A negative
            value indicates (available cpu's + num_dataloader_workers + 1) - same semantics as often used in sklearn.
            e.g., -1 indicates as many workers as cpu's, -2 indicates 1 fewer than the number of cpu's, etc.
        """
        self.gpu_enabled = cuda_available()
        self._torch_module = None
        self._optimizer = None
        self.is_classifier = is_classifier

        # property setter method ensures this goes to the gpu if it's available
        self.torch_module = torch_module

        # property setter method gets the torch.optim class if this is a string, checks inheritance, passes
        # module params and optimizer_kwargs to constructor
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = optimizer

        self.loss_func_kwargs = loss_func_kwargs
        self.loss_func = loss_func
        # you could pass in a logger.info/debug or a file.write method for this if you like
        self.print = print_func
        self.stopping_criterion = stopping_criterion

        self.norm_n_samples = estimate_normalization_samples
        self._input_mean = None
        self._input_sd = None
        self._norm_estimated = False

        self.encode_input = input_encoder
        self.encode_target = target_encoder
        if output_decoder is not None:
            self.decode_output = output_decoder

        # these take tensors and wrap them in Variables and move them to the GPU if necessary
        self.prepare_input = self.get_input_preparer()
        self.prepare_target = self.get_target_preparer()

        self.num_dataloader_workers = num_dataloader_workers

    @property
    def should_normalize(self):
        return self.norm_n_samples is not None

    def estimate_normalization(self, sample: Union[FloatTensorType, ndarray]):
        """Estimate the input normalization parameters (mean and sd) per input dimension and store them for input
        normalization during fitting and prediction"""
        if not self.should_normalize:
            raise ValueError("This model does not require normalization of inputs; inputs may be class labels or "
                             "pre-normalized")
        mean = sample.mean(0)
        sd = sample.std(0)
        self._input_mean = mean.cuda() if self.gpu_enabled else mean.cpu()
        self._input_sd = sd.cuda() if self.gpu_enabled else sd.cpu()
        self._norm_estimated = True

    def normalize(self, X: Union[FloatTensorType, autograd.Variable]):
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

    def get_input_preparer(self) -> Callable[[TensorType], autograd.Variable]:
        if self.should_normalize:
            if self.gpu_enabled:
                def prepare(data: TensorType) -> autograd.Variable:
                    return autograd.Variable(self.normalize(data.cuda()), volatile=not self._torch_module.training)
            else:
                def prepare(data: TensorType) -> autograd.Variable:
                    return autograd.Variable(self.normalize(data.cpu()), volatile=not self._torch_module.training)
        else:
            if self.gpu_enabled:
                def prepare(data: TensorType) -> autograd.Variable:
                    return autograd.Variable(data.cuda(), volatile=not self._torch_module.training)
            else:
                def prepare(data: TensorType) -> autograd.Variable:
                    return autograd.Variable(data.cpu(), volatile=not self._torch_module.training)
        return prepare

    def get_target_preparer(self) -> Callable[[TensorType], autograd.Variable]:
        if self.gpu_enabled:
            def prepare(data: TensorType) -> autograd.Variable:
                return autograd.Variable(data.cuda(), requires_grad=False, volatile=not self._torch_module.training)
        else:
            def prepare(data: TensorType) -> autograd.Variable:
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

    @property
    def loss_func(self):
        return self._loss_func

    @loss_func.setter
    def loss_func(self, loss_func):
        if isinstance(loss_func, str):
            loss_func = getattr(loss, loss_func)
        if isinstance(loss_func, loss._Loss):
            self._loss_func = loss_func
            self.loss_func_kwargs = None
        else:
            try:
                if issubclass(loss_func, loss._Loss):
                    self._loss_func = loss_func(**self.loss_func_kwargs) if self.loss_func_kwargs else loss_func()
            except:
                raise TypeError("`loss_func` must be a torch.nn.loss._Loss class or instance, "
                                "or a string which refers to one by name")

    def set_mode(self, training: bool):
        if self.torch_module.training != training:
            self.torch_module.train(training)

    def _single_batch_train_pass(self, X_batch: TensorType, y_batch: TensorType, optimizer: optim.Optimizer):
        module = self.torch_module
        module.zero_grad()
        optimizer.zero_grad()
        err = self._single_batch_test_pass(X_batch, y_batch)
        err.backward()
        optimizer.step()
        return err

    def _single_batch_test_pass(self, X_batch: TensorType, y_batch: TensorType):
        y_batch = self.prepare_target(y_batch)
        output = self._single_batch_forward_pass(X_batch)
        err = self.loss_func(output, y_batch)
        return err

    def _single_batch_forward_pass(self, X_batch: TensorType):
        X_batch = self.prepare_input(X_batch)
        output = self.torch_module(X_batch)
        return output

    @training_mode(True)
    def fit(self, X: Iterable[T1], y: Iterable[T2], X_test: Opt[Iterable[T1]]=None, y_test: Opt[Iterable[T2]]=None,
            batch_size: int=DEFAULT_BATCH_SIZE, shuffle: bool=False, max_epochs: int=1,
            min_epochs: int=1, min_rel_improvement: float=1e-5, epochs_without_improvement: int=5,
            batch_report_interval: Opt[int]=None, epoch_report_interval: Opt[int]=None):
        """This method fits the *entire* pipeline, including input normalization. Initialization of weight/bias
        parameters in the torch_module is up to you; there is no obvious canonical way to do it here."""

        if self.should_normalize:
            sample, X = peek(X, self.norm_n_samples)
            if self.encode_input:
                sample = [self.encode_input(x) for x in sample]
            sample = stack(sample)
            self.estimate_normalization(sample)

        self.update(X=X, y=y, X_test=X_test, y_test=y_test,
                    batch_size=batch_size, shuffle=shuffle,
                    max_epochs=max_epochs, min_epochs=min_epochs,
                    min_rel_improvement=min_rel_improvement,
                    epochs_without_improvement=epochs_without_improvement,
                    batch_report_interval=batch_report_interval,
                    epoch_report_interval=epoch_report_interval)

    @training_mode(True)
    def update(self, X: Iterable[T1], y: Iterable[T2], X_test: Opt[Iterable[T1]]=None, y_test: Opt[Iterable[T2]]=None,
               batch_size: int=DEFAULT_BATCH_SIZE, shuffle: bool=False, max_epochs: int=1,
               min_epochs: int=1, min_rel_improvement: Opt[float]=1e-5, epochs_without_improvement: int=5,
               batch_report_interval: Opt[int]=None, epoch_report_interval: Opt[int]=None):
        """Update model parameters in light of new data X and y.
        This method handles packaging X and y into a batch iterator of the kind that torch modules expect"""
        assert max_epochs > 0

        data_kw = dict(X_encoder=self.encode_input, y_encoder=self.encode_target,
                       batch_size=batch_size, shuffle=shuffle,
                       num_workers=self.num_dataloader_workers, classifier=self.is_classifier)

        dataset = efficient_batch_iterator(X, y, **data_kw)
        if X_test is not None and y_test is not None:
            test_data = efficient_batch_iterator(X_test, y_test, **data_kw)
        else:
            if X_test is not None or y_test is not None:
                self.print("Warning: test data was provided but either the regressors or the response were omitted")
            test_data = None

        self._update(dataset, test_data, max_epochs=max_epochs, min_epochs=min_epochs, min_rel_improvement=min_rel_improvement,
                     epochs_without_improvement=epochs_without_improvement, batch_report_interval=batch_report_interval,
                     epoch_report_interval=epoch_report_interval)

    @training_mode(True)
    def fit_zipped(self, dataset: Iterable[Tuple[T1, T2]], test_dataset: Opt[Iterable[Tuple[T1, T2]]]=None,
                   batch_size: int = DEFAULT_BATCH_SIZE, max_epochs: int = 1,
                   min_epochs: int = 1, min_rel_improvement: float = 1e-5, epochs_without_improvement: int = 5,
                   batch_report_interval: Opt[int] = None, epoch_report_interval: Opt[int] = None):
        """For fitting to an iterable sequence of pairs, such as may arise in very large streaming datasets from sources
        that don't fit the random access and known-length requirements of a torch.data.Dataset (e.g. a sequence of
        sentences split from a set of text files as might arise in NLP applications.
        Like TorchModel.fit(), this estimates input normalization before the weight update, and weight initialization of
        the torch_module is up to you.
        This method handles packaging X and y into a batch iterator of the kind that torch modules expect"""

        if self.should_normalize:
            sample, dataset = peek(dataset, self.norm_n_samples)
            sample = [t[0] for t in sample]
            if self.encode_input:
                sample = [self.encode_input(x) for x in sample]
            sample = stack(sample)
            self.estimate_normalization(sample)

        self.update_zipped(dataset=dataset, test_dataset=test_dataset, batch_size=batch_size,
                           max_epochs=max_epochs, min_epochs=min_epochs, min_rel_improvement=min_rel_improvement,
                           epochs_without_improvement=epochs_without_improvement,
                           batch_report_interval=batch_report_interval, epoch_report_interval=epoch_report_interval)

    @training_mode(True)
    def update_zipped(self, dataset: Iterable[Tuple[T1, T2]], test_dataset: Opt[Iterable[Tuple[T1, T2]]]=None,
                      batch_size: int = DEFAULT_BATCH_SIZE, max_epochs: int = 1,
                      min_epochs: int = 1, min_rel_improvement: Opt[float] = 1e-5, epochs_without_improvement: int = 5,
                      batch_report_interval: Opt[int] = None, epoch_report_interval: Opt[int] = None):
        """For updating model parameters in light of an iterable sequence of (x,y) pairs, such as may arise in very
        large streaming datasets from sources that don't fit the random access and known-length requirements of a
        torch.data.Dataset (e.g. a sequence of sentences split from a set of text files as might arise in NLP
        applications."""

        data_kw = dict(batch_size=batch_size, classifier=self.is_classifier,
                       X_encoder=self.encode_input,
                       y_encoder=self.encode_target)

        dataset = TupleIteratorDataLoader(dataset, **data_kw)

        if test_dataset is not None:
            test_dataset = TupleIteratorDataLoader(test_dataset, **data_kw)

        self._update(dataset, test_dataset, max_epochs=max_epochs, min_epochs=min_epochs,
                     min_rel_improvement=min_rel_improvement, epochs_without_improvement=epochs_without_improvement,
                     batch_report_interval=batch_report_interval, epoch_report_interval=epoch_report_interval)

    @training_mode(True)
    def fit_dataloader(self, batches: Iterable[Tuple[TensorType, TensorType]],
                       test_batches: Opt[Iterable[Tuple[TensorType, TensorType]]],
                       max_epochs: int = 1, min_epochs: int = 1,
                       min_rel_improvement: Opt[float] = 1e-5, epochs_without_improvement: int = 5,
                       batch_report_interval: Opt[int] = None, epoch_report_interval: Opt[int] = None):
        """For fitting to an iterable of batch tensor pairs, such as would come from a torch.util.data.DataLoader.
        Variables are therefore assumed to be already appropriately encoded, and none of the provided encoders is used.
        The test set is also assumed to be in this form. Like TorchModel.fit(), this estimates input normalization
        before the weight update, and weight initialization of the torch_module is up to you."""

        if self.should_normalize:
            sample = []
            batch_iter = iter(batches)
            n_samples = 0
            while n_samples < self.norm_n_samples:
                batch = next(batch_iter)
                sample.extend(batch)
                n_samples += len(batch)

            sample = stack(sample)
            self.estimate_normalization(sample)

        self._update(batches=batches, test_batches=test_batches, max_epochs=max_epochs, min_epochs=min_epochs,
                     min_rel_improvement=min_rel_improvement, epochs_without_improvement=epochs_without_improvement,
                     batch_report_interval=batch_report_interval, epoch_report_interval=epoch_report_interval)

    def _update(self, batches: Iterable[Tuple[TensorType, TensorType]], test_batches: Opt[Iterable[Tuple[TensorType, TensorType]]],
                max_epochs: int = 1, min_epochs: int = 1, min_rel_improvement: Opt[float] = 1e-5, epochs_without_improvement: int = 5,
                batch_report_interval: Opt[int] = None, epoch_report_interval: Opt[int] = None):
        # all training ultimately ends up here
        if test_batches is None:
            losstype = 'training'
        else:
            losstype = 'validation'

        optimizer = self.optimizer

        epoch, epoch_loss, epoch_time, epoch_samples = 0, 0.0, 0.0, 0
        test_loss, test_samples = None, None
        epoch_losses = deque(maxlen=epochs_without_improvement)

        for epoch in range(max_epochs):
            epoch_start = time()
            if epoch_report_interval and epoch % epoch_report_interval == epoch_report_interval - 1:
                self.print("Training epoch {}".format(epoch + 1))

            running_loss = 0.0
            epoch_samples = 0
            for i, (X_batch, y_batch) in enumerate(batches):
                batch_start = time()
                batch_samples = X_batch.size(0)
                epoch_samples += batch_samples

                err = (self._single_batch_train_pass(X_batch, y_batch, optimizer)).data[0]
                running_loss += err
                batch_time = time() - batch_start

                if batch_report_interval and i % batch_report_interval == batch_report_interval - 1:
                    self.report_batch(epoch, i, err, batch_samples, batch_time)

            epoch_time = time() - epoch_start

            if test_batches is not None:
                test_loss, test_samples = self._error(test_batches)

            if epoch_report_interval and epoch % epoch_report_interval == epoch_report_interval - 1:
                self.report_epoch(epoch, epoch_loss, test_loss, losstype, epoch_samples, test_samples, epoch_time)
                self.print()

            epoch_losses.append(epoch_loss)
            if epoch + 1 >= min_epochs and self.stop_training(epoch_losses, min_rel_improvement):
                break

        if epoch_report_interval:  # and epoch % epoch_report_interval != epoch_report_interval - 1:
            self.report_epoch(epoch, epoch_loss, test_loss, losstype, epoch_samples, test_samples, epoch_time)

    # aliases
    train = fit
    train_zipped = fit_zipped
    train_dataloader = fit_dataloader
    update_dataloader = _update

    def report_epoch(self, epoch: int, epoch_loss: float, test_loss: float, losstype: str, epoch_samples: int,
                     test_samples: int, runtime: float):
        lossname = self.loss_func.__class__.__name__
        test_loss = test_loss or epoch_loss
        test_samples = epoch_loss or epoch_samples
        loss_ = round(test_loss/test_samples, 4)
        self.print("epoch {}, {} samples, {} {} per sample: {}".format(epoch + 1, epoch_samples, losstype, lossname, loss_))
        t, sample_t = pretty_time(runtime), pretty_time(runtime / epoch_samples)
        self.print("Total runtime: {}  Runtime per sample: {}".format(t, sample_t))

    def report_batch(self, epoch: int, batch: int, batch_loss: float, n_samples: int, runtime: float):
        lossname = self.loss_func.__class__.__name__
        sample_t = pretty_time(runtime / n_samples)
        loss = round(batch_loss/n_samples, 4)
        self.print("epoch {}, batch {}, {} samples, runtime per sample: {}, {} per sample: {}"
                   "".format(epoch + 1, batch + 1, n_samples, sample_t, lossname, loss))

    def stop_training(self, epoch_losses: Iterable[float], min_rel_improvement: float) -> bool:
        tup = self.stopping_criterion(list(epoch_losses), min_rel_improvement)
        if isinstance(tup, tuple):
            stop, stop_msg = tup[0:2]
        else:
            stop, stop_msg = tup, None
        if stop:
            if stop_msg:
                self.print(stop_msg)
        return stop

    @training_mode(False)
    def error(self, X: Iterable[T1], y: Iterable[T2], batch_size: int = DEFAULT_BATCH_SIZE, shuffle: bool=False) -> float:
        dataset = efficient_batch_iterator(X, y, X_encoder=self.encode_input, y_encoder=self.encode_target,
                                           batch_size=batch_size, shuffle=shuffle, 
                                           num_workers=self.num_dataloader_workers)
        err, n_samples = self._error(dataset)
        return err / n_samples

    @training_mode(False)
    def error_zipped(self, dataset: Iterable[Tuple[T1, T2]], batch_size: int = DEFAULT_BATCH_SIZE) -> float:
        """For computing per-sample loss on an iterable sequence of (x,y) pairs, such as may arise in very
        large streaming datasets from sources that don't fit the random access and known-length requirements of a
        torch.data.Dataset (e.g. a sequence of sentences split from a set of text files as might arise in NLP
        applications.
        This method handles packaging X and y into a batch iterator of the kind that torch modules expect"""

        data_kw = dict(batch_size=batch_size, classifier=self.is_classifier,
                       X_encoder=self.encode_input,
                       y_encoder=self.encode_target)

        dataset = TupleIteratorDataLoader(dataset, **data_kw)
        err, n_samples = self._error(dataset)
        return err / n_samples

    @training_mode(False)
    def error_dataloader(self, batches: Iterable[Tuple[TensorType, TensorType]]):
        """For computing loss on an iterable of batch tensor pairs, such as would come from a torch.util.data.DataLoader.
        Variables are therefore assumed to be already appropriately encoded, and none of the provided encoders is used.
        """
        err, n_samples = self._error(batches)
        return err / n_samples

    def _error(self, batches: Iterable[Tuple[TensorType, TensorType]]) -> Tuple[float, int]:
        running_loss = 0.0
        running_samples = 0
        for X_batch, y_batch in batches:
            err = self._single_batch_test_pass(X_batch, y_batch)
            running_loss += err.data[0]
            running_samples += X_batch.size()[0]
        return running_loss, running_samples

    # aliases
    loss = error
    loss_zipped = error_zipped
    loss_dataloader = error_dataloader

    @training_mode(False)
    def predict(self, X: Iterable[Any], batch_size: int = DEFAULT_BATCH_SIZE, shuffle: bool=False) -> Iterable[T2]:
        dataset = efficient_batch_iterator(X, X_encoder=self.encode_input, y_encoder=self.encode_input,
                                           batch_size=batch_size, shuffle=shuffle, 
                                           num_workers=self.num_dataloader_workers)
        return self._predict(dataset)

    @training_mode(False)
    def predict_dataloader(self, batches: Iterable[Tuple[TensorType, TensorType]]):
        return self._predict(batches)

    def _predict(self, batches: Iterable[Tuple[TensorType, TensorType]]) -> Iterable[T2]:
        for X_batch, _ in batches:
            for output in self._single_batch_forward_pass(X_batch):
                yield self.decode_output(output)
    
    @staticmethod
    def encode_input(X: T1) -> TensorType:
        """encode the input to a tensor that can be fed to the neural net;
        this can be passed to the class constructor for customizability, else it is assumed to be the identity."""
        return X

    @staticmethod
    def encode_target(y: T2) -> TensorType:
        """encode the output to a tensor that can be used to compute the error of a neural net prediction;
        this can be passed to the class constructor for customizability, else it is assumed to be the identity."""
        return y

    @staticmethod
    def decode_output(y: Iterable[TensorType]) -> T2:
        """take the output Variable from the neural net and decode it to whatever type the training set target was;
        this can be passed to the class constructor for customizability, else it is assumed to be the identity."""
        return y

    def _init_dict(self):
        return dict(loss_func = self.loss_func,
                    loss_func_kwargs = self.loss_func_kwargs,
                    optimizer = self._optimizer.__name__,
                    optimizer_kwargs = self.optimizer_kwargs,
                    input_encoder = self.encode_input,
                    target_encoder = self.encode_target,
                    output_decoder = self.decode_output,
                    is_classifier = self.is_classifier,
                    estimate_normalization_samples = self.norm_n_samples,
                    stopping_criterion = self.stopping_criterion,
                    num_dataloader_workers = self.num_dataloader_workers)

    def _state_dict(self):
        mean, sd = self._input_mean, self._input_sd
        return dict(_input_mean = get_torch_object_bytes(mean) if mean is not None else mean,
                    _input_sd = get_torch_object_bytes(sd) if sd is not None else sd,
                    _norm_estimated = self._norm_estimated)

    def save(self, path: Union[str, IO]):
        state = self.__getstate__()
        with open_file(path, 'wb') as outfile:
            pickle.dump(state, outfile)

    @classmethod
    def load(cls, path: Union[str, IO]) -> 'TorchModel':
        with open_file(path, 'rb') as infile:
            state = pickle.load(infile)
        model = cls.__new__(cls)
        model.__setstate__(state)
        return model

    # for using pickle.dump/load directly
    def __getstate__(self):
        return (self._init_dict(), self._state_dict(), get_torch_object_bytes(self.torch_module))

    def __setstate__(self, state):
        init_dict, state_dict, torch_bytes = state
        module = load_torch_object_bytes(torch_bytes)
        self.__init__(torch_module=module, **init_dict)
        for k, v in state_dict.items():
            value = v if not isinstance(v, bytes) else load_torch_object_bytes(v)
            self.__dict__.__setitem__(k, value)


class TorchSequenceModel(TorchModel):
    def __init__(self, torch_module: nn.Module, loss_func: loss._Loss,
                 optimizer: optim.Optimizer,
                 loss_func_kwargs: Opt[dict]=None,
                 optimizer_kwargs: Opt[dict]=None,
                 input_encoder: Opt[Callable[[T1], TensorType]]=None,
                 target_encoder: Opt[Callable[[T2], TensorType]]=None,
                 output_decoder: Opt[Callable[[TensorType], T2]]=None,
                 clip_grad_norm: Opt[float]=None,
                 is_classifier: bool=False,
                 flatten_targets: bool=True,
                 estimate_normalization_samples: Opt[int]=None,
                 print_func: Callable[[Any], None]=print, num_dataloader_workers: int=-2):
        super(TorchSequenceModel, self).__init__(torch_module=torch_module, loss_func=loss_func, optimizer=optimizer,
                                                 loss_func_kwargs=loss_func_kwargs, optimizer_kwargs=optimizer_kwargs,
                                                 input_encoder=input_encoder,
                                                 target_encoder=target_encoder, output_decoder=output_decoder,
                                                 is_classifier=is_classifier,
                                                 estimate_normalization_samples=estimate_normalization_samples,
                                                 print_func=print_func, num_dataloader_workers=num_dataloader_workers)

        self.flatten_targets = flatten_targets
        self.clip_grad_norm = clip_grad_norm

    @property
    def clip_grad(self):
        return self.clip_grad_norm is not None

    def _single_batch_train_pass(self, X_batch: TensorType, y_batch: TensorType, optimizer: optim.Optimizer):
        module = self.torch_module
        optimizer.zero_grad()
        err = self._single_batch_test_pass(X_batch, y_batch)
        err.backward()
        if self.clip_grad:
            clip_grad_norm(module.parameters(), self.clip_grad_norm)
        optimizer.step()
        return err

    def _single_batch_test_pass(self, X_batch: TensorType, y_batch: TensorType):
        y_batch = self.prepare_target(y_batch)
        if self.flatten_targets:
            y_batch = y_batch.view(-1)
        output = self._single_batch_forward_pass(X_batch)
        err = self.loss_func(output, y_batch)
        return err

    def _single_batch_forward_pass(self, X_batch: TensorType):
        X_batch = self.prepare_input(X_batch)
        output = self.torch_module(X_batch)
        if isinstance(output, tuple):
            output = output[0]
        return output

    def estimate_normalization(self, sample: Union[FloatTensorType, ndarray]):
        if isinstance(sample, FloatTensorTypes):
            sample = sample.numpy()
        sample = sample[0:self.norm_n_samples]
        # statistics along both batch and sequence axes; this functionality is why we need numpy
        mean = from_numpy(sample.mean((0,1)))
        sd = from_numpy(sample.std((0,1)))
        self._input_mean = mean.cuda() if self.gpu_enabled else mean.cpu()
        self._input_sd = sd.cuda() if self.gpu_enabled else sd.cpu()
        self._norm_estimated = True

    # note: TorchModel.normalize should still work since Tensor.expand_as does what we hope it would do

    def _init_dict(self):
        d = super(TorchSequenceModel, self)._init_dict()
        d['flatten_targets'] = self.flatten_targets
        d['clip_grad_norm'] = self.clip_grad_norm
        return d
