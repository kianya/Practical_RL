import torch
import torch.nn as nn
import torch.nn.functional as F

# Note: unlike official pytorch tutorial, this model doesn't process one sample at a time
# because it's slow on GPU.  instead it uses masks just like ye olde theano/tensorflow.
# it doesn't use torch.nn.utils.rnn.pack_paded_sequence because reasons.

"""
encoder---decoder

           P(y|h)
             ^
 LSTM  ->   LSTM
  ^          ^
 biLSTM  -> LSTM
  ^          ^
input       y_prev
"""

class AdvancedTranslationModel(nn.Module):
    def __init__(self, inp_voc, out_voc,
                 emb_size, hid_size,):
        
        super(self.__class__, self).__init__()
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.hid_size = hid_size

        self.emb_inp = nn.Embedding(len(inp_voc), emb_size)
        self.emb_out = nn.Embedding(len(out_voc), emb_size)
        
        self.enc0 = nn.LSTM(emb_size, hid_size, batch_first=True, bidirectional=True)
        self.enc1 = nn.LSTM(hid_size, hid_size, batch_first=True)

        self.dec_start_0 = nn.Linear(hid_size, hid_size)
        self.dec_start_1 = nn.Linear(hid_size, hid_size)
        
        self.dec0 = nn.LSTMCell(emb_size, hid_size)
        self.dec1 = nn.LSTMCell(hid_size*2, hid_size)
        
        self.logits = nn.Linear(hid_size, len(out_voc))

    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: a vector of input tokens  (Variable, int64, 1d)
        :return: a list of initial decoder state tensors
        """
        inp_emb = self.emb_inp(inp)
        enc_seq_0, cx_0 = self.enc0(inp_emb)
        print('enc_seq_0', enc_seq_0.size())
        print('cx_0', cx_0[0][-1].size(), cx_0[1][-1].size())
        
        enc_seq_0 = enc_seq_0[:, :, :self.hid_size] + enc_seq_0[:, :, self.hid_size:]
        enc_seq_1, cx_1 = self.enc1(enc_seq_0)
        print('cx_1', cx_1[0].size(), cx_0[1].size())
        
        # select last element w.r.t. mask
        end_index = infer_length(inp, self.inp_voc.eos_ix)
        end_index[end_index >= inp.shape[1]] = inp.shape[1] - 1

        enc_last_0 = enc_seq_0[range(0, enc_seq_0.shape[0]), end_index.detach(), :]
        print('enc_last_0', enc_last_0.size())
        enc_last_1 = enc_seq_1[range(0, enc_seq_1.shape[0]), end_index.detach(), :]

        dec_start_0 = self.dec_start_0(enc_last_0)
        print('dec_start_0', dec_start_0.size())
        dec_start_1 = self.dec_start_1(enc_last_1)
        return [dec_start_0, cx_0, dec_start_1, cx_1]

    def decode(self, prev_state, prev_tokens, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch,n_tokens]
        """
        [enc_last1, cx1 ,enc_last2, cx2] = prev_state

        prev_emb = self.emb_out(prev_tokens)
        print('prev_emb', prev_emb.size())
        new_dec_state1, cx1 = self.dec0(prev_emb, (cx1[0][-1], cx1[1][-1]))
        new_dec_state2, cx2 = self.dec1(torch.cat((new_dec_state1, enc_last2),1))
        
        output_logits = self.logits(new_dec_state2)

        return [new_dec_state1, cx1, new_dec_state2, cx2], output_logits

    def forward(self, inp, out, eps=1e-30, **flags):
        """
        Takes symbolic int32 matrices of hebrew words and their english translations.
        Computes the log-probabilities of all possible english characters given english prefices and hebrew word.
        :param inp: input sequence, int32 matrix of shape [batch,time]
        :param out: output sequence, int32 matrix of shape [batch,time]
        :return: log-probabilities of all possible english characters of shape [bath,time,n_tokens]

        Note: log-probabilities time axis is synchronized with out
        In other words, logp are probabilities of __current__ output at each tick, not the next one
        therefore you can get likelihood as logprobas * tf.one_hot(out,n_tokens)
        """
        device = next(self.parameters()).device
        batch_size = inp.shape[0]
        bos = torch.tensor(
            [self.out_voc.bos_ix] * batch_size,
            dtype=torch.long,
            device=device,
        )
        logits_seq = [torch.log(to_one_hot(bos, len(self.out_voc)) + eps)]

        hid_state = self.encode(inp, **flags)
        for x_t in out.transpose(0, 1)[:-1]:
            hid_state, logits = self.decode(hid_state, x_t, **flags)
            logits_seq.append(logits)

        return F.log_softmax(torch.stack(logits_seq, dim=1), dim=-1)

    def translate(self, inp, greedy=False, max_len=None, eps=1e-30, **flags):
        """
        takes symbolic int32 matrix of hebrew words, produces output tokens sampled
        from the model and output log-probabilities for all possible tokens at each tick.
        :param inp: input sequence, int32 matrix of shape [batch,time]
        :param greedy: if greedy, takes token with highest probablity at each tick.
            Otherwise samples proportionally to probability.
        :param max_len: max length of output, defaults to 2 * input length
        :return: output tokens int32[batch,time] and
                 log-probabilities of all tokens at each tick, [batch,time,n_tokens]
        """
        device = next(self.parameters()).device
        batch_size = inp.shape[0]
        bos = torch.tensor(
            [self.out_voc.bos_ix] * batch_size,
            dtype=torch.long,
            device=device,
        )
        mask = torch.ones(batch_size, dtype=torch.uint8, device=device)
        logits_seq = [torch.log(to_one_hot(bos, len(self.out_voc)) + eps)]
        out_seq = [bos]
        
        hid_state = self.encode(inp, **flags)
        while True:
            hid_state, logits = self.decode(hid_state, out_seq[-1], **flags)
            if greedy:
                _, y_t = torch.max(logits, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                y_t = torch.multinomial(probs, 1)[:, 0]
            
            logits_seq.append(logits)
            out_seq.append(y_t)
            
            mask &= torch.tensor(y_t != self.out_voc.eos_ix, dtype=torch.uint8)

            if not mask.any():
                break
            if max_len and len(out_seq) >= max_len:
                break

        return (
            torch.stack(out_seq, 1),
            F.log_softmax(torch.stack(logits_seq, 1), dim=-1),
        )


### Utility functions ###
def infer_mask(
        seq,
        eos_ix,
        batch_first=True,
        include_eos=True,
        dtype=torch.float):
    """
    compute length given output indices and eos code
    :param seq: tf matrix [time,batch] if batch_first else [batch,time]
    :param eos_ix: integer index of end-of-sentence token
    :param include_eos: if True, the time-step where eos first occurs is has mask = 1
    :returns: lengths, int32 vector of shape [batch]
    """
    assert seq.dim() == 2
    is_eos = (seq == eos_ix).to(dtype=torch.float)
    if include_eos:
        if batch_first:
            is_eos = torch.cat((is_eos[:, :1] * 0, is_eos[:, :-1]), dim=1)
        else:
            is_eos = torch.cat((is_eos[:1, :] * 0, is_eos[:-1, :]), dim=0)
    count_eos = torch.cumsum(is_eos, dim=1 if batch_first else 0)
    mask = count_eos == 0
    return mask.to(dtype=dtype)


def infer_length(
        seq,
        eos_ix,
        batch_first=True,
        include_eos=True,
        dtype=torch.long):
    """
    compute mask given output indices and eos code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos_ix: integer index of end-of-sentence token
    :param include_eos: if True, the time-step where eos first occurs is has mask = 1
    :returns: mask, float32 matrix with '0's and '1's of same shape as seq
    """
    mask = infer_mask(seq, eos_ix, batch_first, include_eos, dtype)
    return torch.sum(mask, dim=1 if batch_first else 0)


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data
    y_tensor = y_tensor.to(dtype=torch.long).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(
        y_tensor.size()[0],
        n_dims,
        device=y.device,
    ).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot
