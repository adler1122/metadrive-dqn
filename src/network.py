import torch.nn as nn

class DQNNetwork(nn.Module):

    def __init__(self, state_size, hidden_size, action_size=6):
        super(DQNNetwork, self).__init__()
        
        # layer 1: Takes in the 259 game numbers 
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU()
        
        # layer 2:
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        
        # layer 3 (Output): connects the 256/512/128/64 neurons down to our 6 final choices (actions)
        self.fc3 = nn.Linear(hidden_size, action_size)


    def forward(self, x): 
        """
        data 'x' flows through layer 1, then layer 2, and outputs the 6 action scores.
        """
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        q_values = self.fc3(x)
        
        return q_values