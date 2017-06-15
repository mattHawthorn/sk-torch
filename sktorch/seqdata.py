#coding:utf-8

from itertools import repeat, chain
from itertools import takewhile
from operator import itemgetter
from random import sample
from typing import Iterable, List, Dict, Tuple, Iterator, Optional as Opt, Sequence as Seq, Mapping as Map
from torch import stack
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset
# from torch.utils.data.dataloader import default_collate

from .data import H, T1, str_to_int_tensor_type, IntTensorType, FloatTensorType, NumericTensorTypes


#####################################################################
# Sequence data utils                                               #
#####################################################################


class SpecialToken(str):
    def __init__(self, token):
        self.name = self.__class__.__name__
        self = token

    def __str__(self):
        return self

    def __repr__(self):
        return '%s("%s")' % (self.name, self)


OOV = type('OOV', (SpecialToken,), {})
EOS = type('EOS', (SpecialToken,), {})
Null = type('Null', (SpecialToken,), {})
DEFAULT_OOV = "<oov>"
DEFAULT_EOS = "<eos>"
DEFAULT_NULL = "<null>"


class Vocabulary:
    """
    A two-way hash table mapping entities to int IDs and a reverse hash mapping IDs to entities.
    """
    def __init__(self, oov_token: H=DEFAULT_OOV):
        # hash unigram --> ID
        self.token2id = dict()
        # hash ID --> unigram
        self.id2token = dict()

        self.maxID = -1
        if oov_token is not None:
            self.oov_token = oov_token

    @property
    def size(self):
        return len(self.token2id)

    @property
    def oov_token(self):
        return self._oov_token

    @oov_token.setter
    def oov_token(self, oov_token: H):
        oov_token = OOV(oov_token)
        self._oov_token = oov_token
        self.add(oov_token)
        self._oov_id = self.token2id[oov_token]

    @property
    def oov_id(self):
        return self._oov_id

    def add(self, token: H):
        self.add_many((token,))

    def add_many(self, tokens: Iterable[H]):
        for token in tokens:
            if token not in self.token2id:
                # increment the maxID and vocabSize
                self.maxID += 1
                # set both mappings
                self.token2id[token] = self.maxID
                self.id2token[self.maxID] = token

    def __len__(self):
        return len(self.token2id)

    def get_ids(self, tokens: Seq[H]) -> List[int]:
        encoder = self.token2id.get
        oov = self._oov_id
        return [encoder(t, oov) for t in tokens]

    def get_tokens(self, ids: Seq[int]) -> List[H]:
        return list(self.get_tokens_iter(ids))

    def get_tokens_iter(self, ids: Seq[int]) -> Iterator[H]:
        decoder = self.id2token.get
        oov = self._oov_token
        return (decoder(i, oov) for i in ids)

    @classmethod
    def from_token2id(cls, token2id: Dict[H, int], oov_token: H=DEFAULT_OOV):
        return cls.from_token_id_tuples(token2id.items(), oov_token)

    @classmethod
    def from_id2token(cls, id2token: Dict[int, T1], oov_token: H=DEFAULT_OOV):
        tuples = ((token, i) for i, token in id2token.items())
        return cls.from_token_id_tuples(tuples, oov_token)

    @classmethod
    def from_token_id_tuples(cls, token_id_tuples: Iterable[Tuple[T1, int]], oov_token: H=DEFAULT_OOV):
        # don't want to take up the 0 id at the start
        vocab = Vocabulary(oov_token=None)
        token2id = vocab.token2id
        id2token = vocab.id2token
        for token, i in token_id_tuples:
            if token in token2id:
                raise ValueError("Multiple ids for token {}".format(token))
            if i in id2token:
                raise ValueError("Multiple tokens for id {}".format(i))
            token2id[token] = i
            id2token[i] = token
            vocab.maxID = max(vocab.maxID, i)
        vocab.oov_token = oov_token
        return vocab

class SequenceTensorEncoder:
    def __init__(self, vocab: Vocabulary, append_eos: bool = True, eos_token: Opt[H] = DEFAULT_EOS,
                 batch_first: bool = True,
                 pack_sequences: bool=False, null_token: H = DEFAULT_NULL, int_id_type: str = 'long'):
            """Encoder/decoder for going from sequences to tensors and back. The encoder() and decoder() methods
            can be passed to a TorchModel as the input_encoder and output_decoder kwargs. Additionally, the
            collate_batch method can be passed as the collate_fn arg to a DataLoader instance for wrapping up sequences
            as tensors.
            :param vocab: instance of Vocabulary() to use for encoding/decoding tokens.
            :param int_id_type: string indicating the type of int ids to use. Must be a key of data.str_to_int_tensor_type.
            :param pack_sequences: bool indicating whether to return batches as PackedSequences or simply padded tensors.
            :param null_token: Optional hashable to use for padding sequences. Added to the vocab, unless none is passed
                and none is built, in which case this is considered to be an int id.
                Numpy aliases for integer types are valid, as well as 'long', 'short', 'byte', 'char'.
                The default 'long' is recommended, as only LongTensors can be used to index Embeddings in pytorch.

            """
            self.vocab = vocab
            self.tensor_type = str_to_int_tensor_type[int_id_type.lower()]
            self.eos_token = eos_token
            self.null_token = null_token
            self.pack_sequences = pack_sequences
            self.batch_first = batch_first
            self.append_eos = append_eos

    @property
    def eos_token(self):
        return self._eos_token

    @eos_token.setter
    def eos_token(self, eos_token: H):
        if eos_token is not None:
            self._eos_token = EOS(eos_token)
            self.vocab.add(self._eos_token)
            self._eos_id = self.vocab.token2id[self._eos_token]
        else:
            self._eos_token = None
            self._eos_id = None

    @property
    def eos_id(self):
        return self._eos_id

    @property
    def null_token(self):
        return self._null_token

    @null_token.setter
    def null_token(self, null_token: H):
        self._null_token = Null(null_token)
        self.vocab.add(self._null_token)
        self._null_id = self.vocab.token2id[self._null_token]

    @property
    def null_id(self):
        return self._null_id

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, tokens: Seq[H]) -> List[int]:
        ids = self.vocab.get_ids(tokens)
        if self.append_eos:
            ids.append(self._eos_id)
        return ids

    def encode_tensor(self, tokens: Seq[H]) -> IntTensorType:
        return self.tensor_type(self.encode(tokens))

    def pad_encode(self, tokens: Seq[H], length: int) -> List[int]:
        ids = self.vocab.get_ids(tokens)
        if self.append_eos:
            ids.append(self._eos_id)
        padding = repeat(self.null_id, max(0,length - len(tokens)))
        return list(chain(ids, padding))

    def pad_encode_tensor(self, tokens: Seq[H], length: int) -> IntTensorType:
        return self.tensor_type(self.pad_encode(tokens, length))

    def pad_tensor(self, tensor: IntTensorType, length: int) -> IntTensorType:
        l = len(tensor)
        if l >= length:
            return tensor
        else:
            _tensor = tensor.resize_(length)
            _tensor[l:] = self._null_id
            return _tensor

    def decode(self, tensor: IntTensorType) -> IntTensorType:
        # tensor seq of int ids
        tokens = self.vocab.get_tokens_iter(tensor)
        return list(tokens)

    def pad_decode(self, tensor: IntTensorType) -> IntTensorType:
        # tensor seq of int ids
        null = self.null_token
        tokens = takewhile(lambda i: i != null, self.vocab.get_tokens_iter(tensor))
        return list(tokens)

    def decode_preds(self, tensor: FloatTensorType) -> List[H]:
        # tensor.size() = (seq_len, vocab_size)
        vals, ixs = tensor.max(1)
        ids = list(ixs.squeeze())
        return self.vocab.get_tokens(ids)

    def collate_batch(self, batch):
        encode = self.encode_batch if not self.pack_sequences else self.encode_batch_packed
        if isinstance(batch[0], tuple):
            # for the x and y case
            batch = zip(*batch)
            return [encode(x) for x in batch]
        elif isinstance(batch, list):
            return encode(batch)
        else:
            raise TypeError("Unsure how to collate batch; data must be a list of tuples or a list of lists of tokens,"
                            "not: \n{}".format(batch))

    def encode_batch(self, batch: List[Seq[H]]):
        seq_lens = zip(batch, map(len, batch))
        seqs, lens = zip(*seq_lens)
        max_len = max(*lens)
        tensor = self.package_tensor(seqs, max_len)
        return tensor if self.batch_first else tensor.transpose_(0,1)

    def encode_batch_packed(self, batch: List[Seq[H]]):
        seq_lens = sorted(zip(batch, map(len, batch)), key=itemgetter(1), reverse=True)
        seqs, lens = zip(*seq_lens)
        max_len = lens[0]
        tensor = self.package_tensor(seqs, max_len)
        return pack_padded_sequence(tensor, lengths=lens, batch_first=self.batch_first)

    def package_tensor(self, seqs, max_len):
        if isinstance(seqs[0], NumericTensorTypes):
            return stack([self.pad_tensor(t, max_len) for t in seqs])
        else:
            return self.tensor_type([self.pad_encode(tokens, max_len) for tokens in seqs])


class RNNSequencePredictorDataset(Dataset):
    """Subclass of torch.utils.data.Dataset - can be safely passed to a torch.utils.data.DataLoader for multithreaded
    data loading. Assumes a sequence prediction (rather than classification) task; dependent variable is simply the
    independent variable sequence offset by 1."""
    def __init__(self, sequences: Map[int, Seq[H]], encoder: SequenceTensorEncoder, max_len: Opt[int]=None,
                 null_token: H=DEFAULT_NULL, shuffle: bool=True):
        """
        :param sequences: Iterable of sequences of tokens or any other discrete entity
        :param max_len: maximum length of sequences to return in training batches
        :param encoder: instance of SequenceTensorEncoder
        :param shuffle: bool; whether to shuffle examples when iterating over this dataset directly (as opposed to
            using a DataLoader to load batches).
        """
        self.encoder = encoder
        self.encode = encoder.encode
        self.decode = encoder.decode
        self.collate_fn = encoder.collate_batch
        self.null_token = null_token

        self.sequences = sequences
        self.shuffle = shuffle
        self.max_len = max_len

        if self.max_len is not None:
            def ix_pairs(tup):
                i, seq = tup
                addon = 1 if not encoder.append_eos else 2
                return zip(repeat(i), range(max(len(seq) - max_len + addon, 1)))
            self.seq_idxs = list(chain.from_iterable(map(ix_pairs, enumerate(self.sequences))))
        else:
            self.seq_idxs = None

    @property
    def vocab_size(self):
        return self.encoder.vocab_size

    def __len__(self):
        return len(self.seq_idxs) if self.max_len is not None else len(self.sequences)

    def __getitem__(self, idx) -> IntTensorType:
        encode = self.encoder.encode_tensor
        if self.max_len is not None:
            seq_idx, position = self.seq_idxs[idx]
            seq = encode(self.sequences[seq_idx])
            x = seq[position:position + self.max_len]
            y = seq[(position + 1):(position + self.max_len + 1)]
        else:
            seq = encode(self.sequences[idx])
            x, y = seq[0:-1], seq[1:]
        return x, y

    def __iter__(self):
        idxs = range(len(self))
        if self.shuffle:
            idxs = sample(list(idxs), len(idxs))
        getsample = self.__getitem__
        return (getsample(i) for i in idxs)

    @classmethod
    def from_vocab(cls, sequences: Map[int, Seq[H]], vocab: Vocabulary, max_len: int, pack_sequences: bool=False,
                   append_eos: bool=True, eos_token: Opt[H]=DEFAULT_EOS, null_token: H=DEFAULT_NULL,
                   int_id_type: str='long', shuffle: bool=True):
        """
        :param vocab: instance of Vocabulary to use for encoding/decoding tokens
        :param max_len: maximum length of sequences to sample
        :param pack_sequences: bool indicating whether to return regular Tensors or PackedSequence instances.
        :param int_id_type: string indicating the type of int ids to use. Must be a key of data.str_to_int_tensor_type.
        :param eos_token: string or hashable to append to mark end-of-sequence in encoding
        :param null_token: Optional hashable to use for padding sequences. Added to the vocab, unless none is passed
            and none is built, in which case this is considered to be an int id.
            Numpy aliases for integer types are valid, as well as 'long', 'short', 'byte', 'char'.
            The default 'long' is recommended, as only LongTensors can be used to index Embeddings in pytorch.
        """
        encoder = SequenceTensorEncoder(vocab, append_eos=append_eos, eos_token=eos_token, null_token=null_token,
                                        int_id_type=int_id_type)
        return cls(sequences=sequences, encoder=encoder, max_len=max_len, pack_sequences=pack_sequences,
                   null_token=null_token, shuffle=shuffle)

    @classmethod
    def from_token2id(cls, sequences: Map[int, Seq[H]], token2id: Dict[H, int],
                      max_len: int, pack_sequences: bool=False,
                      append_eos: bool=True, eos_token: Opt[H]=DEFAULT_EOS,
                      null_token: H=DEFAULT_NULL, oov_token: H=DEFAULT_OOV,
                      int_id_type: str='long', shuffle: bool=True):
        """
        :param token2id: mapping of tokens to int ids
        :param max_len: maximum length of sequences to sample
        :param pack_sequences: bool indicating whether to return regular Tensors or PackedSequence instances.
        :param int_id_type: string indicating the type of int ids to use. Must be a key of data.str_to_int_tensor_type.
        :param oov_token: hashable to insert for out-of-vocab tokens when encoding
        :param eos_token: string or hashable to append to mark end-of-sequence in encoding
        :param null_token: Optional hashable to use for padding sequences. Added to the vocab, unless none is passed
            and none is built, in which case this is considered to be an int id.
            Numpy aliases for integer types are valid, as well as 'long', 'short', 'byte', 'char'.
            The default 'long' is recommended, as only LongTensors can be used to index Embeddings in pytorch.
        """
        vocab = Vocabulary.from_token2id(token2id, oov_token=oov_token)
        encoder = SequenceTensorEncoder(vocab, append_eos=append_eos, eos_token=eos_token, null_token=null_token,
                                        int_id_type=int_id_type)
        return cls(sequences=sequences, encoder=encoder, max_len=max_len, pack_sequences=pack_sequences,
                   null_token=null_token, shuffle=shuffle)

    @classmethod
    def from_id2token(cls, sequences: Map[int, Seq[H]], id2token: Dict[H, int],
                      max_len: int, pack_sequences: bool=False,
                      append_eos: bool=True, eos_token: Opt[H]=DEFAULT_EOS,
                      null_token: H=DEFAULT_NULL, oov_token: H=DEFAULT_OOV,
                      int_id_type: str='long', shuffle: bool=True):
        """
        :param id2token: mapping of int ids to tokens
        :param max_len: maximum length of sequences to sample
        :param pack_sequences: bool indicating whether to return regular Tensors or PackedSequence instances.
        :param int_id_type: string indicating the type of int ids to use. Must be a key of data.str_to_int_tensor_type.
        :param oov_token: hashable to insert for out-of-vocab tokens when encoding
        :param eos_token: hashable to append to mark end-of-sequence in encoding
        :param null_token: hashable to use for padding sequences. Added to the vocab, unless none is passed
            and none is built, in which case this is considered to be an int id.
            Numpy aliases for integer types are valid, as well as 'long', 'short', 'byte', 'char'.
            The default 'long' is recommended, as only LongTensors can be used to index Embeddings in pytorch.
        """
        vocab = Vocabulary.from_id2token(id2token, oov_token=oov_token)
        encoder = SequenceTensorEncoder(vocab, append_eos=append_eos, eos_token=eos_token, null_token=null_token,
                                        int_id_type=int_id_type)
        return cls(sequences=sequences, encoder=encoder, max_len=max_len, pack_sequences=pack_sequences,
                   null_token=null_token, shuffle=shuffle)