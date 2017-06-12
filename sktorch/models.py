#coding:utf-8

from torch import nn
from torch.nn.utils.rnn import PackedSequence


# RNN Based Language Model
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, oov_id: int, embed_size: int, hidden_size: int, num_layers: int=1,
                 tie_weights: bool=False, batch_first: bool=True,
                 dropout: float=0.0, init_weight_sd: float=0.1,
                 init_forget_gate_bias: float=1.0, init_output_gate_bias: float=1.0):
        if tie_weights and (embed_size != hidden_size):
            raise ValueError("With tied weights, the embedding dimension must equal the hidden state dimension.")

        super(LSTMLanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size + 1, embed_size, padding_idx=vocab_size)
        dropout = 0.0 if not dropout else dropout
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=True)

        self.init_forget_gate_bias = init_forget_gate_bias
        self.init_output_gate_bias = init_output_gate_bias
        self.init_weight_sd = init_weight_sd

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
    def hidden_size(self):
        return self.lstm.hidden_size

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

    def hidden_state(self, x, h=None):
        out, hc = self.lstm_forward(x, h)
        return hc[0]

    def cell_state(self, x, h=None):
        out, hc = self.lstm_forward(x, h)
        return hc[1]

    def hidden_and_cell_state(self, x, h=None):
        out, hc = self.lstm_forward(x, h)
        return hc

    def forward(self, x, h=None):
        out, h = self.lstm_forward(x, h)

        packed = isinstance(out, PackedSequence)

        if packed:
            data = out.data
            out = PackedSequence(self.linear(data))
        else:
            # Reshape output to (batch_size*sequence_length, hidden_size)
            batch, seq_len = out.size(0), out.size(1)
            data = out.contiguous().view(batch * seq_len, self.hidden_size)
            # Decode hidden states of all time steps
            out = self.linear(data).view(batch, seq_len, self.vocab_size)

        return out
