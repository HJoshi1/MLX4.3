#%%
import torch
import torch.nn as nn

#%%
#Analogue of the nn.RNN module
class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh'):
        super(MyRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Initialize parameters
        self.weight_ih = nn.Parameter(torch.Tensor(num_layers, hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(num_layers, hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(num_layers, hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(num_layers, hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        '''
        This function defines a forward RNN pass  

        Input: tensor of shape (batch_size, sequence_length, input_size)'
        Output: (output, hx) where output is a list of tensors oh  cell
        predictions, shape (num_layers, batch_size, hidden_size)
        '''
        # Initializes the hidden state if not provided
        if hx is None:
            hx = torch.zeros(self.num_layers, input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)

        outputs = []

        #iterate over each time step
        for i in range(input.size(1)):
            hx = self.rnn_cell(input[:, i, :], hx)
            outputs.append(hx.unsqueeze(1))

        output = torch.cat(outputs, dim=1)
        return output, hx

    def rnn_cell(self, input, hx):
        '''
        Defines a run of one RNN batch for one time step

        Inputs: 
            input tensor of hape (batch_size, 1, input_size)
            hx tensor of shape (num_layers, batch_size, hidden_size)
        Output:
            tensor of shape (num_layers, batch_size, hidden_size)

        '''
        # Apply RNN cell computation  --> tensor (batch_size, hidden_size)
        gates = torch.matmul(input, self.weight_ih.transpose(0, 1)) + torch.matmul(hx, self.weight_hh.transpose(0, 1))
        if self.bias_ih is not None:
            gates += self.bias_ih.unsqueeze(0)
            gates += self.bias_hh.unsqueeze(0)
        if self.nonlinearity == 'tanh':
            return torch.tanh(gates)
        elif self.nonlinearity == 'relu':
            return torch.relu(gates)
        else:
            raise ValueError("Unsupported nonlinearity. Choose from 'tanh' or 'relu'.")


# %%
class fullRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(fullRNN, self).__init__()
        self.hidden_size=hidden_size
        self.rnn_cell=nn.RNN(input_size, hidden_size)
        self.output_layer=nn.Linear(hidden_size, output_size)
    def forward(self, ordered_text):
        '''
        This functions defines forward prop through our RNN network.
        The input is a tensor of shape (seq_length, batch_size, input_size)
        The seq_length is number of examples
        '''
        batch_size=ordered_text.size(1)
        hidden=self.init_hidden(batch_size)
        rnn_output, hidden = self.rnn_cell(ordered_text, hidden)
        output=self.output_layer(rnn_output[-1, :, :])
        return output
    def init_hidden(self, batch_size):
        '''
        Initiates the hidden layer for the whole text
        (the number of words/tokens is batch_size) at once
        '''
        return torch.zeros(1, batch_size, self.hidden_size)
        
    

# # # %%
# #TEST
# # Example parameters
# input_size = 6  # Size of each input feature
# hidden_size = 10  # Size of the hidden state
# output_size = 4 # Size of the output

# # Create an instance of the SimpleRNN model
# rnn = fullRNN(input_size, hidden_size, output_size)

# # Generate some random input data
# seq_length = 10  # Length of each sequence
# batch_size = 3   # Number of sequences in each batch

# # Generate random input sequence
# input_seq = torch.randn(seq_length, batch_size, input_size)
# print('Input_seq:', input_seq)
# # Forward pass through the RNN model
# output = rnn(input_seq)
# #%%
# rnn.forward(input_seq)

# print("Output shape:", output.shape)
# # %%

# # %%
