from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PeepholeLSTMCell(nn.Module):
    """A single LSTM cell with peephole connections.

    Equations:
      i_t = sigmoid(W_xi x + W_hi h + W_ci * c_prev + b_i)
      f_t = sigmoid(W_xf x + W_hf h + W_cf * c_prev + b_f)
      g_t = tanh(W_xg x + W_hg h + b_g)
      c_t = f_t * c_prev + i_t * g_t
      o_t = sigmoid(W_xo x + W_ho h + W_co * c_t + b_o)
      h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # input to gates
        self.wx = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        # hidden to gates
        self.wh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        # peephole parameters (W_ci, W_cf, W_co) as learnable vectors
        self.w_ci = nn.Parameter(torch.zeros(hidden_size))
        self.w_cf = nn.Parameter(torch.zeros(hidden_size))
        self.w_co = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, input_size], hx,cx: [batch, hidden_size]
        gates = self.wx(x) + self.wh(hx)
        # gates layout: i | f | g | o (each hidden_size)
        i, f, g, o = gates.chunk(4, dim=1)

        # apply peephole: add w_ci * c_prev to input gate, w_cf * c_prev to forget gate
        i = i + cx * self.w_ci
        f = f + cx * self.w_cf

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)

        c_next = f * cx + i * g

        # output gate uses c_next peephole
        o = o + c_next * self.w_co
        o = torch.sigmoid(o)

        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class NbePeepholeLSTM(nn.Module):
    """Peephole LSTM network that maps sequences to sequences.

    Accepts input of shape (batch, seq_len, input_size) and returns
    output of shape (batch, seq_len, output_size). By default output_size==input_size.

    The cell predicts "next time" values given the current input at each timestep.
    Two modes of operation:
      - teacher forcing (default): process the provided input sequence and produce
        predictions for each time step (y_t predicted from x_t).
      - autoregressive (inference): seed with x[:,0,:] and repeatedly feed the
        previous prediction as next input to generate a full sequence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.num_layers = num_layers
        self.dropout = dropout

        # build stack of peephole cells
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            self.cells.append(PeepholeLSTMCell(in_size, hidden_size))

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else None

        # output projection from hidden to output_size
        self.out_proj = nn.Linear(hidden_size, self.output_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rewrite_next: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_size)
            hidden: optional tuple (h0, c0) where each is (num_layers, batch, hidden_size)
            autoregressive: if True, ignore x beyond t=0 and generate sequence autoregressively.

        Returns:
            outputs: (batch, seq_len, output_size)
            (h_n, c_n): final states
        """
        batch, seq_len, _ = x.shape

        # initialize hidden if needed
        if hidden is None:
            h = x.new_zeros((self.num_layers, batch, self.hidden_size))
            c = x.new_zeros((self.num_layers, batch, self.hidden_size))
        else:
            h, c = hidden

        outputs = []
        # If rewrite_next is requested, we'll need a writable copy of x
        if rewrite_next:
            x_work = x.clone()
        else:
            x_work = x

        # teacher-forced processing: for each time step t, process x_work[:,t,:]
        for t in range(seq_len):
            input_t = x_work[:, t, :]
            h_new = []
            c_new = []
            layer_input = input_t
            for layer_idx, cell in enumerate(self.cells):
                hx = h[layer_idx]
                cx = c[layer_idx]
                h_next, c_next = cell(layer_input, hx, cx)
                if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                    layer_input = self.dropout_layer(h_next)
                else:
                    layer_input = h_next
                h_new.append(h_next)
                c_new.append(c_next)
            h = torch.stack(h_new, dim=0)
            c = torch.stack(c_new, dim=0)
            out = self.out_proj(layer_input)
            outputs.append(out.unsqueeze(1))
            # optionally overwrite the beginning of the next input with the produced output
            if rewrite_next and t + 1 < seq_len:
                # determine how many elements to copy (can't exceed input size)
                replace_len = min(self.output_size, self.input_size)
                # out: (batch, output_size) -> take first replace_len
                x_work[:, t + 1, :replace_len] = out[:, :replace_len]
        outputs = torch.cat(outputs, dim=1)

        return outputs, (h, c)

    def predict_next(self, x_t: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Given a single timestep input `x_t` (batch, input_size) and optional hidden states,
        compute the next timestep prediction and return (out, (h,c)). If `hidden` is None,
        initialize hidden states to zeros.
        """
        batch = x_t.shape[0]
        if hidden is None:
            h = x_t.new_zeros((self.num_layers, batch, self.hidden_size))
            c = x_t.new_zeros((self.num_layers, batch, self.hidden_size))
        else:
            h, c = hidden

        layer_input = x_t
        h_new = []
        c_new = []
        for layer_idx, cell in enumerate(self.cells):
            hx = h[layer_idx]
            cx = c[layer_idx]
            h_next, c_next = cell(layer_input, hx, cx)
            if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                layer_input = self.dropout_layer(h_next)
            else:
                layer_input = h_next
            h_new.append(h_next)
            c_new.append(c_next)

        h = torch.stack(h_new, dim=0)
        c = torch.stack(c_new, dim=0)
        out = self.out_proj(layer_input)
        return out, (h, c)


if __name__ == "__main__":
    # quick shape smoke test
    batch = 4
    seq = 19
    inp = 27
    hid = 64
    model = NbePeepholeLSTM(input_size=inp, hidden_size=hid, output_size=inp, num_layers=2, dropout=0.1)
    x = torch.randn(batch, seq, inp)
    y, (hn, cn) = model(x)
    print('y', y.shape)
    y2, _ = model(x, autoregressive=True)
    print('y autoreg', y2.shape)
