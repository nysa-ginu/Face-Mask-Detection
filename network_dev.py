import snntorch as snn
import torch
import torch.nn as nn
import numpy as np


batch_size = 137

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


input_dim = 32*32
hidden_dim = 1000
output_dim = 2

timesteps = 30
b = 0.80

#class for Spiking Neural Network
class SNNNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=b)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=b)

    def forward(self, x):

        mem_poten_1 = self.lif1.init_leaky()
        mem_poten_2 = self.lif2.init_leaky()

        spike_list = []
        mem_list = []

        for step in range(timesteps):
            cur1 = self.linear1(x)
            spike_1, mem_poten_1 = self.lif1(cur1, mem_poten_1)
            cur2 = self.linear2(spike_1)
            spike_2, mem_poten_2 = self.lif2(cur2, mem_poten_2)
            spike_list.append(spike_2)
            mem_list.append(mem_poten_2)

        return torch.stack(spike_list, dim=0), torch.stack(mem_list, dim=0)
        
snn_net = SNNNet().to(device)
    
loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(snn_net.parameters(), lr=5e-4, betas=(0.9, 0.999))