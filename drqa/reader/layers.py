# Here we have created defintions of model layers and NN modules
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Modules are defined as follows"""

class StackedBRNN(nn.Module):
    
  """Class for a Stacked Bi-directional RNNs. This is different from the standard PyTorch library in that it has the option to save and concat the hidden states between layers."""
    
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        
         """Encoding padded or non-padded sequences.
         Choice is available to either handle or ignore sequences of given variable length.
         Padding is always handled in eval.
        Arguemnts:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        
        if x_mask.data.sum() == 0:
         # No padding necessary.

            output = self.forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            
         # Pad if we care or if its during eval.

            output = self.forward_with_padding(x, x_mask)
        else:
            
         # We don't care.

            output = self.forward_unpadded(x, x_mask)

        return output.contiguous()

    def forward_unpadded(self, x, x_mask):
        """Quick encoding that ignores every padding."""

        x = x.transpose(0, 1)
        # Encoding all the layers.
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]
        output = output.transpose(0, 1)
        
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def forward_with_padding(self, x, x_mask):
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        lengths = list(lengths[idx_sort])

        
        x = x.index_select(0, idx_sort)

        
        x = x.transpose(0, 1)

       
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

       
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, padding], 1)

        
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class SeqAttnMatch(nn.Module):
    

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        
       
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Calculating scores.
        scores = x_proj.bmm(y_proj.transpose(2, 1))

       # Padding Mask
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

       # Normalizing with softmax function.

        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=-1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

       
        matched_seq = alpha.bmm(y)
        return matched_seq


class BilinearSeqAttn(nn.Module):
    """Expression for a bilinear attention layer over a sequence X w.r.t y is as follows:
    * o_i = softmax(x_i'Wy) for x_i in X.
    This is optional,it is not necessary to normalize output weights.
    """
    

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize

        # If given identity is true, we just use a dot product without transformation.

        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        
           """
        Arguments:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
       
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize:
            if self.training:
                
                alpha = F.log_softmax(xWy, dim=-1)
            else:
                
                alpha = F.softmax(xWy, dim=-1)
        else:
            alpha = xWy.exp()
        return alpha


class LinearSeqAttn(nn.Module):
    """Formula for computing self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=-1)
        return alpha




def uniformity_in_weights(x, x_mask):
    """Returning uniform weights over non-masked x (a sequence of vectors).
    Arguments:
        x: batch * len * hdim
        x_mask: batch * len (1 for padding, 0 for true)
    Output:
        x_avg: batch * hdim
    """
    
    alpha = torch.ones(x.size(0), x.size(1))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha


def average_weighted(x, weights):
     """Returning a weighted average of x (in a sequence of vectors).
    Arguments:
        x: batch * len * hdim
        weights: batch * len, sum(dim = 1) = 1
    Output:
        x_avg: batch * hdim
    """
    
    return weights.unsqueeze(1).bmm(x).squeeze(1)
