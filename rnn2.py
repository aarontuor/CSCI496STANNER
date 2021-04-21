"""
Recurrent Neural Network implementation for use with structured linear maps.
"""
# machine learning/data science imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# ecosystem imports
import slim

#mnist data loading and dependencies for Steven's implementation
import idx2numpy
import numpy as np
import argparse


#2

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=False, nonlin=F.gelu,
                 hidden_map=slim.Linear, input_map=slim.Linear, input_args=dict(),
                 hidden_args=dict()):
        """

        :param input_size: (int) Dimension of input to rnn cell.
        :param hidden_size: (int) Dimension of output of rnn cell.
        :param bias: (bool) Whether to use bias.
        :param nonlinearity: (callable) Activation function
        :param linear_map: (nn.Module) A module compatible with torch.nn.Linear
        :param linargs: (dict) Arguments to instantiate linear layers

        """
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.in_features, self.out_features = input_size, hidden_size
        self.nonlin = nonlin
        self.lin_in = input_map(input_size, hidden_size, bias=bias, **input_args)
        self.lin_hidden = hidden_map(hidden_size, hidden_size, bias=bias, **hidden_args)

    def reg_error(self):
        """

        :return: (torch.float) Regularization error associated with linear maps.
        """
        return (self.lin_in.reg_error() + self.lin_hidden.reg_error())/2.0

    def forward(self, input, hidden):
        """

        :param input: (torch.Tensor, shape=[batch_size, input_size]) Input to cell
        :param hidden: (torch.Tensor, shape=[batch_size, hidden_size]) Hidden state (typically previous output of cell)
        :return: (torch.Tensor, shape=[batchsize, hidden_size]) Cell output

        .. doctest::

            >>> import slim, torch
            >>> cell = slim.RNNCell(5, 8, input_map=slim.Linear, hidden_map=slim.PerronFrobeniusLinear)
            >>> x, h = torch.rand(20, 5), torch.rand(20, 8)
            >>> output = cell(x, h)
            >>> output.shape
            torch.Size([20, 8])

        """

        return self.nonlin(self.lin_hidden(hidden) + self.lin_in(input))


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=1, cell_args=dict()):
        """
        Has input and output corresponding to basic usage of torch.nn.RNN module.
        No bidirectional, bias, nonlinearity, batch_first, and dropout args.
        Cells can incorporate custom linear maps.
        Bias and nonlinearity are included in cell args.

        :param input_size: (int) Dimension of inputs
        :param hidden_size: (int) Dimension of hidden states
        :param num_layers: (int)  Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together
                                  to form a stacked RNN, with the second RNN taking in outputs of the first RNN and
                                  computing the final results. Default: 1
        :param cell_args: (dict) Arguments to instantiate RNN cells (see :class:`rnn.RNNCell` for args).
        """
        super().__init__()
        rnn_cells = [RNNCell(input_size, hidden_size, **cell_args)]
        rnn_cells += [RNNCell(hidden_size, hidden_size, **cell_args)
                      for k in range(num_layers-1)]
        self.rnn_cells = nn.ModuleList(rnn_cells)
        self.init_states = nn.ParameterList([nn.Parameter(torch.zeros(1, cell.hidden_size))
                                             for cell in self.rnn_cells])

    def reg_error(self):
        """

        :return: (torch.float) Regularization error associated with linear maps.
        """
        return torch.mean(torch.stack([cell.reg_error() for cell in self.rnn_cells]))

    def forward(self, sequence, init_states=None):
        """
        :param sequence: (torch.Tensor, shape=[seq_len, batch, input_size]) Input sequence to RNN
        :param init_state: (torch.Tensor, shape=[num_layers, batch, hidden_size]) :math:`h_0`, initial hidden states for stacked RNNCells
        :returns:

            - output: (seq_len, batch, hidden_size) Sequence of outputs
            - :math:`h_n`: (num_layers, batch, hidden_size) Final hidden states for stack of RNN cells.

        .. doctest::

            >>> import slim, torch
            >>> rnn = slim.RNN(5, hidden_size=8, num_layers=3, cell_args={'hidden_map': slim.PerronFrobeniusLinear})
            >>> x = torch.rand(20, 10, 5)
            >>> output, h_n = rnn(x)
            >>> output.shape, h_n.shape
            (torch.Size([20, 10, 8]), torch.Size([3, 10, 8]))

        """
        assert len(sequence.shape) == 3, f'RNN takes order 3 tensor with shape=(seq_len, nsamples, {self.insize})'
        if init_states is None:
            init_states = self.init_states
        final_hiddens = []
        for h, cell in zip(init_states, self.rnn_cells):
            # loop over stack of cells
            states = []
            for seq_idx, cell_input in enumerate(sequence):
                # loop over sequence
                h = cell(cell_input, h)
                states.append(h)
            sequence = torch.stack(states)
            final_hiddens.append(h)  # Save final hidden state for each cell in case need to do truncated back prop
        assert torch.equal(sequence[-1, :, :], final_hiddens[-1])
        return sequence, torch.stack(final_hiddens)

def parse_all_args():
    #Parse command line args

    parser = argparse.ArgumentParser()

    parser.add_argument("-lr",type=float,help="learning rate (float) [default: 0.0001]",default=0.0001)
    parser.add_argument("-epochs",type=int,help="Number of training epochs (int) [default: 100]",default=100)
    parser.add_argument("-bs",type=int,help="Batch size (int) [default: 16]",default=16)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_all_args()

    training_data = idx2numpy.convert_from_file('../slim/data/train-images-idx3-ubyte')
    training_labels = idx2numpy.convert_from_file('../slim/data/train-labels-idx1-ubyte')
    test_data = idx2numpy.convert_from_file('../slim/data/t10k-images-idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('../slim/data/t10k-labels-idx1-ubyte')

    training_labels = training_labels.reshape((training_labels.shape[0], 1))
    training_data = training_data.reshape((60000, 784))
    training_data = np.hstack((training_data, training_labels))
    np.random.shuffle(training_data)
    training_labels = training_data[:, -1]
    training_data = training_data[:, :-1]
    training_data = training_data.reshape((60000, 784, 1))
    training_data = torch.from_numpy(training_data).permute(1, 0, 2).numpy()
    validation = training_data[:, 0:10000, :]
    training_data = training_data[:, 10000:, :]

    validation_labels = training_labels[0:10000]
    training_labels = training_labels[10000:]

    # print(validation_labels.shape)
    # print(training_labels.shape)

    # print(validation.shape)
    # print(training_data.shape)

    criterion = torch.nn.NLLLoss()
    # criterion = torch.nn.CrossEntropyLoss()

    # transform data into pytorch float tensors
    training_data = torch.from_numpy(training_data).type(torch.FloatTensor)
    training_labels = torch.from_numpy(training_labels).type(torch.FloatTensor)
    validation = torch.from_numpy(validation).type(torch.FloatTensor)
    validation_labels = torch.from_numpy(validation_labels).type(torch.FloatTensor)

    model = slim.RNN(1, hidden_size = 10, num_layers = 2)

    learning_rate = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    batch_size = args.bs

    for epoch in range(args.epochs):
        for batch in range(0, 50000, batch_size):
            if batch + batch_size >= 50000:
                break
            print(f"training size: {training_data[batch:batch+batch_size, :].shape}")
            model.train()
            model.zero_grad()
            temp = training_data[batch:batch+batch_size, :]
            output = model(training_data[batch:batch+batch_size, :])
            true_labels = training_labels[batch:batch+batch_size]
            loss = criterion(F.log_softmax(output, dim=1), true_labels.type(torch.LongTensor))
            loss.backward()
            optimizer.step()

            if (batch/batch_size)%20:
                pred = torch.argmax(output, dim=1)
                #print(pred, true_labels)
                acc = torch.eq(pred, true_labels).float().mean()
                #validation test
                output= model(validation)
                pred = torch.argmax(output, dim=1)
                dev_acc = torch.eq(pred, validation_labels).float().mean()
                with torch.no_grad():
                    model.eval()
                    dev_loss = criterion(F.log_softmax(output, dim=1), validation_labels.type(torch.LongTensor))

        print(f"train loss&acc: {loss.item()} {acc.item()}\tvalidation loss&acc: {dev_loss.item()} {dev_acc.item()}")
    print("done")

    """
     def __init__(self, input_size, hidden_size=16, num_layers=1, cell_args=dict()):
     
    :param sequence: (torch.Tensor, shape=[seq_len, batch, input_size]) Input sequence to RNN
    :param init_state: (torch.Tensor, shape=[num_layers, batch, hidden_size]) :math:`h_0`, initial hidden states for stacked RNNCells
    :returns:

        - output: (seq_len, batch, hidden_size) Sequence of outputs
        - :math:`h_n`: (num_layers, batch, hidden_size) Final hidden states for stack of RNN cells.

    .. doctest::

        >>> import slim, torch
        >>> rnn = slim.RNN(5, hidden_size=8, num_layers=3, cell_args={'hidden_map': slim.PerronFrobeniusLinear})
        >>> x = torch.rand(20, 10, 5)
        >>> output, h_n = rnn(x)
        >>> output.shape, h_n.shape
        (torch.Size([20, 10, 8]), torch.Size([3, 10, 8]))

            """


    # for bias in [True, False]:
    #     for num_layers in [1, 2]:
    #         for name, map in slim.maps.items():
    #             print(name)
    #             print(map)
    #             rnn = RNN(8, hidden_size=8, num_layers=num_layers, cell_args={'bias': bias,
    #                                                                           'nonlin': F.gelu,
    #                                                                           'hidden_map': map,
    #                                                                           'input_map': slim.Linear})
    #             out = rnn(x)
    #             print(out[0].shape, out[1].shape)
    #
    #         for map in set(slim.maps.values()) - slim.square_maps:
    #             print(name)
    #             rnn = RNN(8, hidden_size=16, num_layers=num_layers, cell_args={'bias': bias,
    #                                                                            'nonlin': F.gelu,
    #                                                                            'hidden_map': map,
    #                                                                            'input_map': slim.Linear})
    #             out = rnn(x)
    #             print(out[0].shape, out[1].shape)
