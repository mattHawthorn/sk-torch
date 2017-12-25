#coding:utf-8

from typing import Union, Optional as Opt
from numpy import ndarray
from torch import nn, sigmoid, log
from torch.nn.utils.rnn import PackedSequence
from .data import NegativeSampler
from calm.processor import ngramIter


class BilinearFactorModel(nn.Module, NegativeSampler):
    def __init__(self, n_input_classes: int, output_dist: Union[int, ndarray],
                 embedding_dim: int, n_neg_samlples: int = 5,
                 neg_sampling_exponent: Opt[float] = None):
        nn.Module.__init__(self)
        NegativeSampler.__init__(self, output_dist = output_dist, n_neg_samlples = n_neg_samlples,
                                 neg_sampling_exponent = neg_sampling_exponent)

        assert isinstance(embedding_dim, int) and embedding_dim > 0, "`embedding_dim` must be a positive int"
        self.embedding_dim = embedding_dim
        assert isinstance(n_input_classes, int) and n_input_classes > 0, "`n_input_classes` must be a positive int"

        self.input_embeddings = nn.Embedding(n_input_classes, self.embedding_dim, sparse=True)
        self.output_embeddings = nn.Embedding(self.n_output_classes, self.embedding_dim, sparse=True)
        self.NCELoss = NCELoss()

    @property
    def n_output_classes(self):
        return len(self.output_dist)

    @property
    def n_input_classes(self):
        return self.input_embeddings.num_embeddings

    def forward(self, in_out_pairs):
        # input is batch_size*2 int Variable
        i = self.input_embeddings(in_out_pairs[:, 0])
        o = self.output_embeddings(in_out_pairs[:, 1])
        # raw activations, NCE_Loss handles the sigmoid (we need to know classes to know the sign to apply)
        return (i * o).sum(1).squeeze()


class NCELoss(nn.Module):
    def forward(self, activations, targets):
        # targets are -1.0 or 1.0, 1-d Variable
        # likelihood assigned by the model to pos and neg samples is given by the sigmoid, with the sign
        # determined by the class.
        # negative log likelihood
        return log(sigmoid(activations * targets)).sum() * -1.0


class LSTMMixin(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_lstm_layers: int = 1,
                 batch_first: bool = True, lstm_dropout: float = 0.0,
                 init_forget_gate_bias: float = 1.0, init_output_gate_bias: float = 1.0):
        super(LSTMMixin, self).__init__()
        lstm_dropout = 0.0 if not lstm_dropout else lstm_dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_lstm_layers, batch_first=batch_first, dropout=lstm_dropout)
        self.init_forget_gate_bias = init_forget_gate_bias
        self.init_output_gate_bias = init_output_gate_bias
        self.batch_first = batch_first
        self.init_lstm_weights()

    def init_lstm_weights(self):
        fg_bias = self.init_forget_gate_bias
        o_bias = self.init_output_gate_bias
        if fg_bias is not None or o_bias is not None:
            h = self.lstm.hidden_size
            lstm_state = self.lstm.state_dict()
            for l in range(self.lstm.num_layers):
                bias_hh, bias_ih = lstm_state['bias_hh_l' + str(l)], lstm_state['bias_ih_l' + str(l)]
                if fg_bias is not None:
                    bias_hh[h:2 * h] = fg_bias
                    bias_ih[h:2 * h] = fg_bias
                if o_bias is not None:
                    bias_hh[3 * h:4 * h] = o_bias
                    bias_ih[3 * h:4 * h] = o_bias

    @property
    def num_lstm_layers(self):
        return self.lstm.num_layers

    @property
    def lstm_dropout(self):
        return self.lstm.dropout

    @property
    def hidden_size(self):
        return self.lstm.hidden_size

    @property
    def input_size(self):
        return self.lstm.input_size

    def lstm_forward(self, x, h=None):
        # override this if you need to reshape/transform inputs before going into the lstm
        return self.lstm(x, h)

    def hidden_state(self, x, h=None):
        out, hc = self.lstm_forward(x, h)
        return hc[0]

    def cell_state(self, x, h=None):
        out, hc = self.lstm_forward(x, h)
        return hc[1]

    def hidden_and_cell_state(self, x, h=None):
        out, hc = self.lstm_forward(x, h)
        return hc


# RNN Based Language Model
class LSTMAutoregressionModel(LSTMMixin):
    def __init__(self, input_size: int, hidden_size: int, num_lstm_layers: int=1,
                 batch_first: bool=True, lstm_dropout: float=0.0, init_weight_sd: float=0.1,
                 init_forget_gate_bias: float=1.0, init_output_gate_bias: float=1.0):
        super(LSTMAutoregressionModel, self).__init__(input_size=input_size, hidden_size=hidden_size,
                                                      num_lstm_layers=num_lstm_layers, batch_first=batch_first,
                                                      lstm_dropout=lstm_dropout,
                                                      init_forget_gate_bias=init_forget_gate_bias,
                                                      init_output_gate_bias=init_output_gate_bias)
        # lstm_dropout = 0.0 if not lstm_dropout else lstm_dropout
        # self.lstm = nn.LSTM(embed_size, hidden_size, num_lstm_layers, batch_first=batch_first, dropout=lstm_dropout)
        self.linear = nn.Linear(hidden_size, input_size, bias=True)

        # self.init_forget_gate_bias = init_forget_gate_bias
        # self.init_output_gate_bias = init_output_gate_bias
        self.init_weight_sd = init_weight_sd
        # self.batch_first = batch_first

        self.init_weights()


    # def init_lstm_weights(self):
    #     fg_bias = self.init_forget_gate_bias
    #     o_bias = self.init_output_gate_bias
    #     if fg_bias is not None or o_bias is not None:
    #         h = self.lstm.hidden_size
    #         lstm_state = self.lstm.state_dict()
    #         for l in range(self.lstm.num_layers):
    #             bias_hh, bias_ih = lstm_state['bias_hh_l' + str(l)], lstm_state['bias_ih_l' + str(l)]
    #             if fg_bias is not None:
    #                 bias_hh[h:2 * h] = fg_bias
    #                 bias_ih[h:2 * h] = fg_bias
    #             if o_bias is not None:
    #                 bias_hh[3 * h:4 * h] = o_bias
    #                 bias_ih[3 * h:4 * h] = o_bias

    def init_weights(self):
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.normal_(0.0, self.init_weight_sd)
        self.init_lstm_weights()

    # def lstm_forward(self, x, h=None):
    #     # Embed word ids to vectors
    #     packed = isinstance(x, PackedSequence)
    #
    #     v = self.vocab_size
    #     data = x if not packed else x.data
    #     if self.oov_id is not None:
    #         data[data >= v] = self.oov_id
    #         data[data < 0] = self.oov_id
    #
    #     if packed:
    #         x = PackedSequence(self.embeddings(data), x.batch_sizes)
    #     else:
    #         x = self.embeddings(data)
    #
    #     # Forward propagate RNN
    #     out, hc = self.lstm(x, h)
    #     return out, hc
    #
    # def hidden_state(self, x, h=None):
    #     out, hc = self.lstm_forward(x, h)
    #     return hc[0]
    #
    # def cell_state(self, x, h=None):
    #     out, hc = self.lstm_forward(x, h)
    #     return hc[1]
    #
    # def hidden_and_cell_state(self, x, h=None):
    #     out, hc = self.lstm_forward(x, h)
    #     return hc

    def forward(self, x, h=None):
        out, h = self.lstm_forward(x, h)

        if isinstance(out, PackedSequence):
            data = out.data
            out = PackedSequence(self.linear(data), out.batch_sizes)
        else:
            # Reshape output to (batch_size*sequence_length, hidden_size)
            # this block works as intended whether batch_first is True or False, though the var names are deceptive
            batch, seq_len = out.size(0), out.size(1)
            data = out.contiguous().view(batch * seq_len, self.hidden_size)
            # Decode hidden states of all time steps
            out = self.linear(data).view(batch, seq_len, self.input_size)

        return out


# RNN Based Language Model
class LSTMLanguageModel(LSTMMixin):
    def __init__(self, vocab_size: int, oov_id: int, embed_size: int, hidden_size: int, num_layers: int=1,
                 tie_weights: bool=False, batch_first: bool=True,
                 dropout: float=0.0, init_weight_sd: float=0.1,
                 init_forget_gate_bias: float=1.0, init_output_gate_bias: float=1.0):
        if tie_weights and (embed_size != hidden_size):
            raise ValueError("With tied weights, the embedding dimension must equal the hidden state dimension.")

        super(LSTMLanguageModel, self).__init__(input_size=embed_size, hidden_size=hidden_size,
                                                num_lstm_layers=num_layers, batch_first=batch_first,
                                                lstm_dropout=dropout, init_forget_gate_bias=init_forget_gate_bias,
                                                init_output_gate_bias=init_output_gate_bias)
        self.embeddings = nn.Embedding(vocab_size + 1, embed_size, padding_idx=vocab_size)
        # dropout = 0.0 if not dropout else dropout
        # self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=True)

        # self.init_forget_gate_bias = init_forget_gate_bias
        # self.init_output_gate_bias = init_output_gate_bias
        self.init_weight_sd = init_weight_sd
        # self.batch_first = batch_first

        if oov_id is not None and 0 < oov_id >= vocab_size:
            raise ValueError("oov_id must be between 0 and vocab_size, exclusive")
        self.oov_id = oov_id
        self.tied = tie_weights
        self.init_weights()

    @property
    def num_layers(self):
        return self.lstm.num_layers

    @property
    def dropout(self):
        return self.lstm.dropout

    @property
    def embed_size(self):
        return self.embeddings.embedding_dim

    @property
    def vocab_size(self):
        return self.linear.out_features

    def init_weights(self):
        self.embeddings.weight.data.normal_(0.0, self.init_weight_sd)
        self.linear.bias.data.fill_(0)

        if self.tied:
            self.linear.weight = self.embeddings.weight
        else:
            self.linear.weight.data.normal_(0.0, self.init_weight_sd)

        self.init_lstm_weights()
        # fg_bias = self.init_forget_gate_bias
        # o_bias = self.init_output_gate_bias
        # if fg_bias is not None or o_bias is not None:
        #     h = self.lstm.hidden_size
        #     lstm_state = self.lstm.state_dict()
        #     for l in range(self.lstm.num_layers):
        #         bias_hh, bias_ih = lstm_state['bias_hh_l' + str(l)], lstm_state['bias_ih_l' + str(l)]
        #         if fg_bias is not None:
        #             bias_hh[h:2 * h] = fg_bias
        #             bias_ih[h:2 * h] = fg_bias
        #         if o_bias is not None:
        #             bias_hh[3 * h:4 * h] = o_bias
        #             bias_ih[3 * h:4 * h] = o_bias

    def lstm_forward(self, x, h=None):
        # Embed word ids to vectors
        packed = isinstance(x, PackedSequence)

        v = self.vocab_size
        data = x if not packed else x.data
        if self.oov_id is not None:
            data[data >= v] = self.oov_id
            data[data < 0] = self.oov_id

        if packed:
            x = PackedSequence(self.embeddings(data), x.batch_sizes)
        else:
            x = self.embeddings(data)

        # Forward propagate RNN
        out, hc = self.lstm(x, h)
        return out, hc

    def forward(self, x, h=None):
        out, h = self.lstm_forward(x, h)

        if isinstance(out, PackedSequence):
            data = out.data
            out = PackedSequence(self.linear(data), out.batch_sizes)
        else:
            # Reshape output to (batch_size*sequence_length, hidden_size)
            # this block works as intended whether batch_first is True or False, though the var names are deceptive
            batch, seq_len = out.size(0), out.size(1)
            data = out.contiguous().view(batch * seq_len, self.hidden_size)
            # Decode hidden states of all time steps
            out = self.linear(data).view(batch, seq_len, self.vocab_size)

        return out
