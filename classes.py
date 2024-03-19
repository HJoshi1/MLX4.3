#%%
import torch
import torch.nn as nn
# %%
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size=hidden_size
        self.rnn_cell=nn.RNN(input_size, hidden_size)
        self.output_layer=nn.Linear(hidden_size, output_size)
    def forward(self, ordered_text):
        '''
        This functions defines forward prop through our RNN network.
        The input is a tensor of shpe (seq_length, batch_size, input_size)
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
        
    

# # %%
# #TEST
# # Example parameters
# input_size = 6  # Size of each input feature
# hidden_size = 10  # Size of the hidden state
# output_size = 4 # Size of the output

# # Create an instance of the SimpleRNN model
# rnn = RNN(input_size, hidden_size, output_size)

# # Generate some random input data
# seq_length = 10  # Length of each sequence
# batch_size = 3   # Number of sequences in each batch

# # Generate random input sequence
# input_seq = torch.randn(seq_length, batch_size, input_size)
# print('Input_seq:', input_seq)
# # Forward pass through the RNN model
# output = rnn(input_seq)

# print("Output shape:", output.shape)
# # %%
