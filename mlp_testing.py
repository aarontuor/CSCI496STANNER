"""
Recurrent Neural Network implementation for use with structured linear maps.
"""
# machine learning/data science imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# ecosystem imports
import slim

# mnist data loading and dependencies for Steven's implementation
import idx2numpy
import numpy as np
import argparse
from slim.bench import *
import matplotlib.pyplot as plt
import time
#2

class MLP(nn.Module):
    """
    Multi-Layer Perceptron consistent with blocks interface
    """
    def __init__(
        self,
        insize,
        outsize,
        bias=True,
        linear_map=slim.Linear,
        nonlin=nn.ReLU,
        hsizes=[64],
        linargs=dict(),
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param bias: (bool) Whether to use bias
        :param linear_map: (class) Linear map class from slim.linear
        :param nonlin: (callable) Elementwise nonlinearity which takes as input torch.Tensor and outputs torch.Tensor of same shape
        :param hsizes: (list of ints) List of hidden layer sizes
        :param linargs: (dict) Arguments for instantiating linear layer
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.nhidden = len(hsizes)
        sizes = [insize] + hsizes + [outsize]
        self.nonlin = [nonlin() for k in range(self.nhidden)] + [nn.Identity()]
        self.linear = nn.ModuleList(
            [
                linear_map(sizes[k], sizes[k + 1], bias=bias, **linargs)
                for k in range(self.nhidden + 1)
            ]
        )

    def reg_error(self):
        return sum([k.reg_error() for k in self.linear if hasattr(k, "reg_error")])

    def forward(self, x):
        """

        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        for lin, nlin in zip(self.linear, self.nonlin):
            x = nlin(lin(x))
        return x

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
    parser.add_argument("-data",type=str,help="Dataset used. Options: mnist, ackley, add, mul, xor (str) [default: mnist]",default="mnist")
    parser.add_argument("-samples",type=int,help="Samples for dataset used if applicable (int) [default: 50000]",default=50000)
    parser.add_argument("-dim",type=int,help="Dimensions for dataset used if applicable (int) [default: 2]",default=2)
    parser.add_argument("-split",type=int,help="Percent of data for training set as a decimal if applicable (float) [default: 0.8]",default=0.8)
    parser.add_argument("-param",type=str,help="Parameterization used in the model. Options: butterfly, pf, linear (str) [default: linear]", default="linear")
    parser.add_argument("-trial",type=int,help="Number in trial if applicable (int) [default: 1]", default=1)

    return parser.parse_args()

def get_mnist(args):
    training_data = idx2numpy.convert_from_file('./data/train-images-idx3-ubyte')
    training_labels = idx2numpy.convert_from_file('./data/train-labels-idx1-ubyte')
    test_data = idx2numpy.convert_from_file('./data/t10k-images-idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('./data/t10k-labels-idx1-ubyte')

    #setup training/validation data
    training_labels = training_labels.reshape((training_labels.shape[0], 1))
    training_data = training_data.reshape((60000, 784))
    training_data = np.hstack((training_data, training_labels))
    np.random.shuffle(training_data)
    training_labels = training_data[:, -1]
    training_data = training_data[:, :-1]
    validation = training_data[0:10000, :]
    training_data = training_data[10000:, :]

    validation_labels = training_labels[0:10000]
    training_labels = training_labels[10000:]

    # transform data into pytorch float tensors
    training_data = torch.from_numpy(training_data).type(torch.FloatTensor)
    training_labels = torch.from_numpy(training_labels).type(torch.FloatTensor)
    validation = torch.from_numpy(validation).type(torch.FloatTensor)
    validation_labels = torch.from_numpy(validation_labels).type(torch.FloatTensor)

    return training_data, training_labels, validation, validation_labels

def get_ackley(args):
    ackley_set = ackley(np.random, args.samples, args.dim)

    X = torch.from_numpy(ackley_set[0])
    y = torch.from_numpy(ackley_set[1])

    num_training_samples = int(args.samples * args.split)

    X_train = X[:num_training_samples, :]
    y_train = y[:num_training_samples]
    X_validation = X[num_training_samples:, :]
    y_validation = y[num_training_samples:]

    # returns training and validation data in tensors
    # X dimensions are [args.samples, args.dim]
    # y dimensions are [args.samples]
    # each is split according to the train/validation ratio given by args.split
    return X_train, y_train, X_validation, y_validation

def get_add(args):
    #args.dim = sequence length
    add_set = add(np.random, args.dim, args.samples)
    X = torch.from_numpy(add_set[0])
    y = torch.from_numpy(add_set[1])

    num_training_samples = int(args.samples * args.split)

    X_train = X[:num_training_samples, :, :]
    y_train = y[:num_training_samples]
    X_validation = X[num_training_samples:, :, :]
    y_validation = y[num_training_samples:]

    #returns training and validation data in tensors
    #X dimensions are [args.samples, args.dim * 2]
    #y dimensions are [args.samples]
    #each is split according to the train/validation ratio given by args.split
    return X_train.view(X_train.shape[0], args.dim * 2), y_train, X_validation.view(X_validation.shape[0], args.dim * 2), y_validation

def get_multiply(args):
    # args.dim = sequence length
    multiply_set = multiply(np.random, args.dim, args.samples)
    X = torch.from_numpy(multiply_set[0])
    y = torch.from_numpy(multiply_set[1])

    num_training_samples = int(args.samples * args.split)

    X_train = X[:num_training_samples, :, :]
    y_train = y[:num_training_samples]
    X_validation = X[num_training_samples:, :, :]
    y_validation = y[num_training_samples:]

    # returns training and validation data in tensors
    # X dimensions are [args.samples, args.dim * 2]
    # y dimensions are [args.samples]
    # each is split according to the train/validation ratio given by args.split
    return X_train.view(X_train.shape[0], args.dim * 2), y_train, X_validation.view(X_validation.shape[0], args.dim * 2), y_validation

def get_xor(args):
    # args.dim = sequence length
    xor_set = xor(np.random, args.dim, args.samples)
    X = torch.from_numpy(xor_set[0])
    y = torch.from_numpy(xor_set[1])

    num_training_samples = int(args.samples * args.split)

    X_train = X[:num_training_samples, :, :]
    y_train = y[:num_training_samples]
    X_validation = X[num_training_samples:, :, :]
    y_validation = y[num_training_samples:]

    # returns training and validation data in tensors
    # X dimensions are [args.samples, args.dim * 2]
    # y dimensions are [args.samples]
    # each is split according to the train/validation ratio given by args.split
    return X_train.view(X_train.shape[0], args.dim * 2), y_train, X_validation.view(X_validation.shape[0], args.dim * 2), y_validation

def grid_search_experiments(args):
    parameterizations = [slim.ButterflyLinear, slim.SpectralLinear, slim.PerronFrobeniusLinear]
    num_layers = [1, 2, 4, 8, 16, 32]
    h_sizes = [8, 16, 32, 64, 128, 256, 512]
    learning_rates = [0.3, 0.1, 0.03, 0.01, 0.003]
    task_datasets = [get_ackley(args), get_add(args), get_multiply(args), get_xor(args)]
    num_models_each = 10
    epochs = 1000
    criterion = torch.nn.MSELoss()

    performance = np.full((len(task_datasets), len(parameterizations), len(num_layers), len(h_sizes), len(learning_rates), num_models_each), -1, dtype=np.float64) #-1 acc = not tested
    best_models = []
    for i in range(len(task_datasets)):
        dim1 = []
        for j in range(len(parameterizations)):
            dim2 = []
            for k in range(len(num_layers)):
                dim3 = []
                for l in range(len(h_sizes)):
                    dim4 = []
                    for m in range(len(learning_rates)):
                        dim5 = []
                        for n in range(num_models_each):
                            dim6 = dict()
                            dim5.append(dim6)
                        dim4.append(dim5)
                    dim3.append(dim4)
                dim2.append(dim3)
            dim1.append(dim2)
        best_models.append(dim1)
    best_models = np.asarray(best_models, dtype=dict)

    for dataset_idx, dataset in enumerate(task_datasets): # 5 datasets
        X_train, y_train, X_val, y_val = dataset
        for param_idx, param in enumerate(parameterizations): # 3 params
            for layers_idx, layers in  enumerate(num_layers): # fix this at 3 layers
                for h_size_idx, h_size in enumerate(h_sizes): # fix at 100
                    hidden_layers = []
                    for i in range(layers):
                        hidden_layers.append(h_size)
                    for lr_idx, lr in enumerate(learning_rates): # pick one or two
                        for trial in range(num_models_each): #num_models_each different random weight initializations
                            #train a model with the given hyperparameters, save the best performing epoch and store its accuracy
                            model = MLP(
                                X_train.shape[1],
                                1,
                                bias=True,
                                linear_map=param,
                                nonlin=nn.ReLU,
                                hsizes=hidden_layers,
                                linargs=dict())

                            num_train_samples = y_train.shape[0]
                            num_val_samples = y_val.shape[0]

                            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                            batch_size = args.bs

                            #start with saving untrained model
                            output = model(X_val)
                            error = criterion(torch.squeeze(output), y_val)
                            print(f"error.item() {error.item()}")
                            performance[dataset_idx, param_idx, layers_idx, h_size_idx, lr_idx, trial] = error.item()
                            best_models[dataset_idx, param_idx, layers_idx, h_size_idx, lr_idx, trial] = model.state_dict()

                            for epoch in range(epochs):
                                for batch in range(0, num_train_samples, batch_size):
                                    if batch + batch_size >= num_train_samples:
                                        break
                                    model.train()
                                    optimizer.zero_grad()
                                    output = model(X_train[batch:batch + batch_size, :])
                                    true_labels = y_train[batch:batch + batch_size]
                                    loss = criterion(torch.squeeze(output), true_labels)
                                    loss.backward()
                                    optimizer.step()

                                #save model and performance if it is the best performing so far
                                model.eval()
                                output = torch.squeeze(model(X_val))
                                error = criterion(output, y_val)
                                print(f"val error: {error.item()}")
                                print(f"samples: \n{output[0]}\t{y_val[0]}\n{output[1]}\t{y_val[1]}")

                                if error.item() < performance[dataset_idx, param_idx, layers_idx, h_size_idx, lr_idx, trial]:
                                    performance[dataset_idx, param_idx, layers_idx, h_size_idx, lr_idx, trial] = error.item()
                                    best_models[dataset_idx, param_idx, layers_idx, h_size_idx, lr_idx, trial] = model.state_dict()

    np.save("grid_search_performance", performance)
    np.save("model_state_dicts", best_models)

def run_mnist(args):
    training_data, training_labels, validation, validation_labels = get_mnist(args)
    criterion = torch.nn.NLLLoss()

    #setup model and hyperparameters
    model = MLP(
        784,
        10,
        bias=True,
        linear_map=slim.PerronFrobeniusLinear,
        nonlin=nn.ReLU,
        hsizes=[128, 128, 128],
        linargs=dict())

    learning_rate = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    batch_size = args.bs

    for epoch in range(args.epochs):
        for batch in range(0, 50000, batch_size):
            if batch + batch_size >= 50000:
                break
            model.train()
            model.zero_grad()
            temp = training_data[batch:batch + batch_size, :]
            output = model(training_data[batch:batch + batch_size, :])
            true_labels = training_labels[batch:batch + batch_size]
            loss = criterion(F.log_softmax(output, dim=1), true_labels.type(torch.LongTensor))
            loss.backward()
            optimizer.step()

            # if (batch/batch_size)%20:
        pred = torch.argmax(output, dim=1)
        # print(pred, true_labels)
        acc = torch.eq(pred, true_labels).float().mean()
        # validation test
        output = model(validation)
        pred = torch.argmax(output, dim=1)
        dev_acc = torch.eq(pred, validation_labels).float().mean()
        with torch.no_grad():
            model.eval()
            dev_loss = criterion(F.log_softmax(output, dim=1), validation_labels.type(torch.LongTensor))

        print(f"train loss&acc: {loss.item()} {acc.item()}\tvalidation loss&acc: {dev_loss.item()} {dev_acc.item()}")
    print("done")

def map_comparison(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device}")
    maps = [slim.ButterflyLinear, slim.Linear, slim.PerronFrobeniusLinear] 
    map_names = []
    for map_type in maps:
        a = map_type(1, 1)
        map_names.append(f"{a.__class__.__name__}")
    training_data, training_labels, validation, validation_labels = get_mnist(args)
    training_data.to(device)
    training_labels.to(device)
    validation.to(device)
    validation_labels.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    #number of maps x number of epochs (num times val performance is measured) x num performance metrics (cross entropy, accuracy)
    #-1 is the placeholder value so it's easy to see what data wasn't aquired
    performance = torch.full((len(maps), args.epochs, 2), -1.0, dtype=torch.float)
    best_performance = torch.full((len(maps), 2), -1.0, dtype=torch.float)
    best_models = []
    for map_type in maps:
        best_models.append(dict())
    best_models = np.asarray(best_models, dtype=dict)

    for map_idx, map_type in enumerate(maps):
        # setup model and hyperparameters
        model = MLP(
            784,
            10,
            bias=True,
            linear_map=map_type,
            nonlin=nn.ReLU,
            hsizes=[512, 512, 256],
            linargs=dict()).to(device)

        best_models[map_idx] = model.state_dict()

        learning_rate = args.lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        batch_size = args.bs
        num_datapoints = training_data.shape[0]

        for epoch in range(args.epochs):
            print(f"Map {map_idx}\tEpoch {epoch}")
            for batch in range(0, num_datapoints, batch_size):
                if batch + batch_size >= num_datapoints:
                    break
                model.train()
                model.zero_grad()
                output = model(training_data[batch:batch + batch_size, :].to(device))
                true_labels = training_labels[batch:batch + batch_size]
                loss = criterion(F.log_softmax(output, dim=1).to(device), true_labels.type(torch.LongTensor).to(device))
                loss.backward()
                optimizer.step()

            # validation test
            output = model(validation.to(device))
            pred = torch.argmax(output, dim=1)
            val_acc = torch.eq(pred.to(device), validation_labels.to(device)).float().mean()
            with torch.no_grad():
                model.eval()
                val_loss = criterion(F.log_softmax(output, dim=1).to(device), validation_labels.type(torch.LongTensor).to(device))
            print(f"validation loss&acc: {val_loss.item()}\t{val_acc.item()}")

            #save epoch performance
            performance[map_idx, epoch, 0] = val_loss.item()
            performance[map_idx, epoch, 1] = val_acc

            #save model and its performance if it had the best accuracy
            if val_acc > best_performance[map_idx, 1]:
                best_performance[map_idx, 0] = val_loss.item()
                best_performance[map_idx, 1] = val_acc
                best_models[map_idx] = model.state_dict()

    np.save("map_comparison/best_performance", np.asarray(best_performance))
    np.save("map_comparison/performance", np.asarray(performance))
    np.save("map_comparison/model_state_dicts", best_models)

    #print best performance
    print(f"\nModel validation performance with best validation accuracy:")
    for map_idx, map_type in enumerate(maps):
        a = map_type(1, 1)
        print(f"{a.__class__.__name__}\tCEL: {best_performance[map_idx, 0]}\tAccruacy: {best_performance[map_idx, 1]}")

    epoch_nums = torch.zeros((args.epochs,), dtype=torch.int32)
    for idx, elem in enumerate(epoch_nums):
        epoch_nums[idx] = idx

    #graph loss
    plt.interactive(False)
    for map_idx, map_type in enumerate(maps):
        plt.plot(epoch_nums, torch.squeeze(performance[map_idx, :, 0]), label = map_names[map_idx])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Cross entropy loss")
    plt.title("Cross Entropy Loss of Different Maps on MNIST")
    plt.legend()
    plt.show()

    #graph accuracy
    for map_idx, map_type in enumerate(maps):
        plt.plot(epoch_nums, torch.squeeze(performance[map_idx, :, 1]), label = map_names[map_idx])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Acuracy of Different Maps on MNIST")
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.show()

def ackley_map_comparison(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device}")
    maps = [slim.Linear, slim.ButterflyLinear, slim.PerronFrobeniusLinear]
    map_names = []
    for map_type in maps:
        a = map_type(1, 1)
        map_names.append(f"{a.__class__.__name__}")
    training_data, training_labels, validation, validation_labels = get_ackley(args)
    training_data.to(device).type(torch.float64)
    training_labels.to(device).type(torch.float64)
    validation.to(device).type(torch.float64)
    validation_labels.to(device).type(torch.float64)

    #normalize input and labels
    #normalize funct: (x-xmin)/(xmax-xmin)
    #denormalize funct: (xnorm)(xmax-xmin)+xmin
    input_min = -40
    input_max = 40

    output_max = torch.max(training_labels)
    output_min = torch.min(training_labels)

    normalization_params = [input_min, input_max, output_min, output_max]

    training_data = torch.subtract(training_data, input_min) / (input_max - input_min)
    training_labels = torch.subtract(training_labels, output_min) / (output_max - output_min)
    validation = torch.subtract(validation, input_min) / (input_max - input_min)
    validation_labels = torch.subtract(validation_labels, output_min) / (output_max - output_min)

    criterion = torch.nn.MSELoss()

    #number of maps x number of epochs (num times val performance is measured) x num performance metrics (MSE, accuracy)
    #-1 is the placeholder value so it's easy to see what data wasn't aquired
    performance = torch.full((len(maps), args.epochs, 2), -1.0, dtype=torch.float)
    best_performance = torch.full((len(maps), 2), -1.0, dtype=torch.float)
    best_models = []
    for map_type in maps:
        best_models.append(dict())
    best_models = np.asarray(best_models, dtype=dict)

    for map_idx, map_type in enumerate(maps):
        # setup model and hyperparameters
        model = MLP(
            2,
            1,
            bias=False,
            linear_map=map_type,
            nonlin=nn.ReLU,
            hsizes=[256, 256, 128],
            linargs=dict()).to(device).type(torch.float64)

        best_models[map_idx] = model.state_dict()

        learning_rate = args.lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        batch_size = args.bs
        num_datapoints = training_data.shape[0]

        for epoch in range(args.epochs):
            print(f"Map {map_idx}\tEpoch {epoch}")
            for batch in range(0, num_datapoints, batch_size):
                if batch + batch_size >= num_datapoints:
                    break
                model.train()
                model.zero_grad()
                output = model(training_data[batch:batch + batch_size, :].to(device))
                true_labels = training_labels[batch:batch + batch_size]
                loss = criterion(output.to(device).squeeze(), true_labels.to(device).squeeze())
                loss.backward()
                optimizer.step()

            # validation test
            output = model(validation.to(device))
            pred = output
            val_acc = torch.eq(pred.to(device), validation_labels.to(device)).float().mean()
            with torch.no_grad():
                model.eval()
                val_loss = criterion(output.to(device).squeeze(), validation_labels.to(device).squeeze())
            print(f"validation loss&acc: {val_loss.item()}\t{val_acc.item()}")

            #save epoch performance
            performance[map_idx, epoch, 0] = val_loss.item()
            performance[map_idx, epoch, 1] = val_acc

            #save model and its performance if it had the best accuracy
            if val_loss.item() < best_performance[map_idx, 0]:
                best_performance[map_idx, 0] = val_loss.item()
                best_performance[map_idx, 1] = val_acc
                best_models[map_idx] = model.state_dict()

    np.save("map_comparison/best_performance", np.asarray(best_performance))
    np.save("map_comparison/performance", np.asarray(performance))
    np.save("map_comparison/model_state_dicts", best_models)
    np.save("map_comparison/normalization_params", np.asarray(normalization_params))

    #print best performance
    print(f"\nModel validation performance with best validation accuracy:")
    for map_idx, map_type in enumerate(maps):
        a = map_type(1, 1)
        print(f"{a.__class__.__name__}\tCEL: {best_performance[map_idx, 0]}\tAccruacy: {best_performance[map_idx, 1]}")

    epoch_nums = torch.zeros((args.epochs,), dtype=torch.int32)
    for idx, elem in enumerate(epoch_nums):
        epoch_nums[idx] = idx

    #graph loss
    plt.interactive(False)
    for map_idx, map_type in enumerate(maps):
        plt.plot(epoch_nums, torch.squeeze(performance[map_idx, :, 0]), label = map_names[map_idx])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Cross entropy loss")
    plt.title("Cross Entropy Loss of Different Maps on MNIST")
    plt.legend()
    plt.show()

    #graph accuracy
    for map_idx, map_type in enumerate(maps):
        plt.plot(epoch_nums, torch.squeeze(performance[map_idx, :, 1]), label = map_names[map_idx])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Acuracy of Different Maps on MNIST")
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.show()


def load_trained_model(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device}")
    maps = [slim.Linear, slim.ButterflyLinear, slim.PerronFrobeniusLinear]
    map_names = []
    for map_type in maps:
        a = map_type(1, 1)
        map_names.append(f"{a.__class__.__name__}")
    training_data, training_labels, validation, validation_labels = get_ackley(args)
    training_data.to(device).type(torch.float64)
    training_labels.to(device).type(torch.float64)
    validation.to(device).type(torch.float64)
    validation_labels.to(device).type(torch.float64)
    
    saved_models = np.load("map_comparison/Ackley_Norm/model_state_dicts.npy", allow_pickle=True)
    normalization_params = [-40, 40, 3.0611, 4600300000000000000]
    performance = np.load("map_comparison/Ackley_Norm/performance.npy", allow_pickle=True)

    #number of maps x number of epochs (num times val performance is measured) x num performance metrics (cross entropy, accuracy)
    #-1 is the placeholder value so it's easy to see what data wasn't aquired
    #print best performance
    print(f"\nModel validation performance with best validation accuracy:")
    for map_idx, map_type in enumerate(maps):
        a = map_type(1, 1)
        print(f"{a.__class__.__name__}\tCEL: {torch.min(torch.from_numpy(performance[map_idx, :, 0]).squeeze())}")

    epoch_nums = torch.zeros((args.epochs,), dtype=torch.int32)
    for idx, elem in enumerate(epoch_nums):
        epoch_nums[idx] = idx

    #graph loss
    plt.interactive(False)
    for map_idx, map_type in enumerate(maps):
        plt.plot(epoch_nums, torch.squeeze(torch.from_numpy(performance[map_idx, :, 0])), label = map_names[map_idx])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Cross entropy loss")
    plt.title("Cross Entropy Loss of Different Maps on MNIST")
    plt.legend()
    plt.show()

    #graph accuracy
    for map_idx, map_type in enumerate(maps):
        plt.plot(epoch_nums, torch.squeeze(torch.from_numpy(performance[map_idx, :, 1])), label = map_names[map_idx])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Acuracy of Different Maps on MNIST")
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.show()

    #normalize data
    training_data_norm = torch.subtract(training_data, normalization_params[0]) / (normalization_params[1] - normalization_params[0])
    training_labels_norm = torch.subtract(training_labels, normalization_params[2]) / (normalization_params[3] - normalization_params[2])
    validation_norm = torch.subtract(validation, normalization_params[0]) / (normalization_params[1] - normalization_params[0])
    validation_labels_norm = torch.subtract(validation_labels, normalization_params[2]) / (normalization_params[3] - normalization_params[2])

    criterion = torch.nn.MSELoss()

    for map_idx, map_type in enumerate(maps):
        if 1 == 1:
            print(f"map type: {map_type}")
            model = MLP(
                    2,
                    1,
                    bias=False,
                    linear_map=map_type,
                    nonlin=nn.ReLU,
                    hsizes=[256, 256, 128],
                    linargs=dict()).to(device).type(torch.float64)

        
            model.load_state_dict(saved_models[map_idx])

            #test loaded model
            #output = model(validation.to(device))
            #pred = output
            #val_acc = torch.eq(pred.to(device), validation_labels.to(device)).float().mean()
            #with torch.no_grad():
            #    model.eval()
            #    val_loss = criterion(output.to(device).squeeze(), validation_labels.to(device).squeeze())
            #print(f"validation loss&acc: {val_loss.item()}\t{val_acc.item()}")

            #make graphs for all maps modelsfig = plt.figure(figsize=(4,4))
            #ax = fig.add_subplot(111, projection='3d')
            #ax.scatter(2,3,4) # plot the point (2,3,4) on the figure
            #plt.show()

            #all_outputs = torch.zeros(validation.shape[0], requires_grad=False)

            #for i in range(validation.shape[0]):
            #    input = validation[i, :].to(device).view(1, -1)
            #    all_outputs[i] = model(input)

            all_outputs = torch.zeros(500, requires_grad=False)

            for i in range(500):
                input = validation_norm[i, :].to(device).view(1, -1)
                all_outputs[i] = model(input)

            #denormalize output
            #denormalize funct: (xnorm)(xmax-xmin)+xmin
            all_outputs = torch.add((all_outputs * (normalization_params[3] - normalization_params[2])), normalization_params[2])

            #find and output denormalized loss
            loss = criterion(all_outputs.to(device), validation_labels[:500].type(torch.LongTensor).to(device))
            print(f"validation denormalized loss on best model: {loss.item()}")

            #make map graphs
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            xdata = validation[:500, 0].squeeze()
            ydata = validation[:500, 1].squeeze()
            zdata = torch.from_numpy(all_outputs.detach().numpy())
    
            ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
            #ax.set_xlim3d(-30, 30)
            #ax.set_ylim3d(-30, 30)
            plt.show()
        
    #make ackley graph
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xdata = validation[:, 0].squeeze()
    ydata = validation[:, 1].squeeze()
    zdata = validation_labels
    
    xdata = training_data[:, 0].squeeze()
    ydata = training_data[:, 1].squeeze()
    zdata = training_labels

    #for idx, xelement in enumerate(xdata):
    #    if xelement > 20:
    #        xdata = torch.cat([xdata[:idx], xdata[idx+1:]])

    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    #ax.set_xlim3d(-30, 30)
    #ax.set_ylim3d(-30, 30)
    plt.show()

def cluster_basic_test(args):
    print("cluster basic test start\n")
    print(f"cuda: {torch.cuda.is_available()}\n")

def cluster_runtime_test(args):
    #example models will run for 10 epochs (time should be multiplied by 100 for 1000epoch real training estimate)
    #one example for each of these maps: linear, perron frobenius, butterfly linear
    #run on the average hyperparameters:
    #  8 layers, 128 nodes per layer
    #after training each model, print the number of epochs and time elapsed
    
    print(f"=====Cluster time to train test=====")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    maps = [slim.Linear, slim.ButterflyLinear, slim.PerronFrobeniusLinear]
    map_names = []
    for map_type in maps:
        a = map_type(1, 1)
        map_names.append(f"{a.__class__.__name__}")
    training_data, training_labels, validation, validation_labels = get_ackley(args)
    training_data.to(device).type(torch.float64)
    training_labels.to(device).type(torch.float64)
    validation.to(device).type(torch.float64)
    validation_labels.to(device).type(torch.float64)

    #normalize input and labels
    #normalize funct: (x-xmin)/(xmax-xmin)
    #denormalize funct: (xnorm)(xmax-xmin)+xmin
    input_min = -40
    input_max = 40

    output_max = torch.max(training_labels)
    output_min = torch.min(training_labels)

    normalization_params = [input_min, input_max, output_min, output_max]

    training_data = torch.subtract(training_data, input_min) / (input_max - input_min)
    training_labels = torch.subtract(training_labels, output_min) / (output_max - output_min)
    validation = torch.subtract(validation, input_min) / (input_max - input_min)
    validation_labels = torch.subtract(validation_labels, output_min) / (output_max - output_min)

    criterion = torch.nn.MSELoss()

    #number of maps x number of epochs (num times val performance is measured) x num performance metrics (MSE, accuracy)
    #-1 is the placeholder value so it's easy to see what data wasn't aquired
    performance = torch.full((len(maps), args.epochs, 2), -1.0, dtype=torch.float)
    best_performance = torch.full((len(maps), 2), -1.0, dtype=torch.float)
    best_models = []
    for map_type in maps:
        best_models.append(dict())
    best_models = np.asarray(best_models, dtype=dict)

    print(f"Each model has 8 layers of 128 nodes. lr = {args.lr}, epochs = {args.epochs}\n")
    for map_idx, map_type in enumerate(maps):
        start = time.time()
        print(f"Starting training for {map_names[map_idx]}")
        # setup model and hyperparameters
        model = MLP(
            2,
            1,
            bias=False,
            linear_map=map_type,
            nonlin=nn.ReLU,
            hsizes=[128, 128, 128, 128, 128, 128, 128, 128],
            linargs=dict()).to(device).type(torch.float64)

        learning_rate = args.lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        batch_size = args.bs
        num_datapoints = training_data.shape[0]

        for epoch in range(args.epochs):
            for batch in range(0, num_datapoints, batch_size):
                if batch + batch_size >= num_datapoints:
                    break
                model.train()
                model.zero_grad()
                output = model(training_data[batch:batch + batch_size, :].to(device))
                true_labels = training_labels[batch:batch + batch_size]
                loss = criterion(output.to(device).squeeze(), true_labels.to(device).squeeze())
                loss.backward()
                optimizer.step()

            # validation test
            output = model(validation.to(device))
            pred = output
            val_acc = torch.eq(pred.to(device), validation_labels.to(device)).float().mean()
            with torch.no_grad():
                model.eval()
                val_loss = criterion(output.to(device).squeeze(), validation_labels.to(device).squeeze())
        end = time.time()
        print(f"Finished training for {map_names[map_idx]}")
        print(f"Epochs: {args.epochs}\tTime to train: {end - start}\n")
   
def train_model_with_params(args):
    print(f"data: {args.data} param: {args.param} trial: {args.trial}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    criterion = torch.nn.MSELoss()
    
    output_size = 1
    X_train, y_train, X_val, y_val = get_ackley(args)
    if(args.data == "add"):
        X_train, y_train, X_val, y_val = get_add(args)
    elif(args.data == "mul"):
        X_train, y_train, X_val, y_val = get_multiply(args)
    elif(args.data == "xor"):
        X_train, y_train, X_val, y_val = get_xor(args)
    elif(args.data == "mnist"):
        X_train, y_train, X_val, y_val = get_mnist(args)
        output_size = 10
        criterion = torch.nn.NLLLoss()
    
    X_train.to(device)
    y_train.to(device)
    X_val.to(device)
    y_val.to(device)
    if(args.data != "mnist"):
	    X_train.type(torch.float64)
	    y_train.type(torch.float64)
	    X_val.type(torch.float64)
	    y_val.type(torch.float64)
		    
								     
     
    if(args.data == "xor"):
        print(f"xor x size: {X_train.shape} and {X_val.shape}\nxor y size: {y_train.shape} and {y_val.shape}")
        
        
     
    param = slim.Linear
    if(args.param == "pf"):
        param = slim.PerronFrobeniusLinear
    elif(args.param == "butterfly"):
        param = slim.ButterflyLinear
        
    num_layers = 3
    num_nodes_per_layer = 100
    lr = 0.001
    epochs = 1000
    #print(f"num_layers: {num_layers} num_nodes_per: {num_nodes_per_layer} lr: {lr} ep: {epochs}\n")
	
	#train model with args
    #train a model with the given hyperparameters, save the best performing epoch and store its accuracy
    model = MLP(
        X_train.shape[1],
        output_size,
        bias=True,
        linear_map=param,
        nonlin=nn.ReLU,
        hsizes=[100, 100, 100],
        linargs=dict()).to(device)#.type(torch.float64)
    if(args.data != "mnist"):
        model.type(torch.float64)

    num_train_samples = y_train.shape[0]
    num_val_samples = y_val.shape[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    batch_size = args.bs

    #start with saving untrained model
    if(args.data != "mnist"):
        output = torch.squeeze(model(X_val.to(device)))
        error = criterion(output.to(device), y_val.to(device))
    else:
        output = model(X_val.to(device))
        error = criterion(F.log_softmax(output, dim=1).to(device), y_val.type(torch.LongTensor).to(device))
    #print(f"error.item() {error.item()}")
    performance = error.item()
    best_model = model.state_dict()

    for epoch in range(epochs):
        for batch in range(0, num_train_samples, batch_size):
            if batch + batch_size >= num_train_samples:
                break
            model.train()
            optimizer.zero_grad()
            output = model(X_train[batch:batch + batch_size, :].to(device))
            true_labels = y_train[batch:batch + batch_size]
            #loss = criterion(torch.squeeze(output).to(device), true_labels.to(device))
            if(args.data != "mnist"):
                loss = criterion(output.to(device).type(torch.float64).squeeze(), true_labels.to(device).type(torch.float64).squeeze())
            else:
                loss = criterion(F.log_softmax(output, dim=1).to(device), true_labels.type(torch.LongTensor).to(device))
            loss.backward()
            optimizer.step()

        #save model and performance if it is the best performing so far
        model.eval()
        if(args.data != "mnist"):
	        output = torch.squeeze(model(X_val.to(device)))
	        error = criterion(output.to(device), y_val.to(device)).to(device)
        else:
            output = model(X_val.to(device))
            error = criterion(F.log_softmax(output, dim=1).to(device), y_val.type(torch.LongTensor).to(device))
        #print(f"val error: {error.item()}")
        #print(f"samples: \n{output[0]}\t{y_val[0]}\n{output[1]}\t{y_val[1]}")

        if error.item() < performance:
            performance = error.item()
            best_model = model.state_dict()

    #save model and performance after last training step
    last_model = model.state_dict()
    model.eval()
    if(args.data != "mnist"):
        output = torch.squeeze(model(X_val.to(device)))
        error = criterion(output.to(device), y_val.to(device))
    else:
        output = model(X_val.to(device))
        error = criterion(F.log_softmax(output, dim=1).to(device), y_val.type(torch.LongTensor).to(device))
    #print(f"last val error: {error.item()}")
    #print(f"samples: \n{output[0]}\t{y_val[0]}\n{output[1]}\t{y_val[1]}")

    last_performance = error.item()
    
    #send best and last models and performances to file
    f = open(f"results/best_performance_{args.data}_{args.param}_{args.trial}.txt", "w")
    f.write(f"data: {args.data}\nparam: {args.param}\ntrial: {args.trial}\nBest performance: {performance}")
    f.close()
    f = open(f"results/last_performance_{args.data}_{args.param}_{args.trial}.txt", "w")
    f.write(f"data: {args.data}\nparam: {args.param}\ntrial: {args.trial}\nLast performance: {last_performance}")
    f.close()
    
    #np.save(f"results/best_performance_{args.data}_{args.param}_{args.trial}", performance)
    np.save(f"models/best_model_{args.data}_{args.param}_{args.trial}", best_model)
    #np.save(f"results/last_performance_{args.data}_{args.param}_{args.trial}", last_performance)
    np.save(f"models/last_model_{args.data}_{args.param}_{args.trial}", last_model)

    print("finished training, saving results, and saving model")

if __name__ == '__main__':
    args = parse_all_args()

    train_model_with_params(args)

    #cluster_basic_test(args)
    
    #cluster_runtime_test(args)

    #ackley_map_comparison(args)

    #load_trained_model(args)

    #grid_search_experiments(args)


    #
    # if args.data == 'mnist':
    #     run_mnist(args)
    # elif args.data == 'ackley':
    #     X_train, y_train, X_validation, y_validation = get_ackley(args)
    # elif args.data == 'add':
    #     X_train, y_train, X_validation, y_validation = get_add(args)
    # elif args.data == 'multiply':
    #     X_train, y_train, X_validation, y_validation = get_multiply(args)
    # elif args.data == 'xor':
    #     X_train, y_train, X_validation, y_validation = get_xor(args)
    #
    # input_size = X_train.shape[1]
    # output_size = y_train[0].shape[0]
    #
    # # setup model and hyperparameters
    # model = MLP(
    #     input_size,
    #     output_size,
    #     bias=True,
    #     linear_map=slim.PerronFrobeniusLinear,
    #     nonlin=nn.ReLU,
    #     hsizes=[128, 128, 128],
    #     linargs=dict())
    #
    # learning_rate = args.lr
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # batch_size = args.bs



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
