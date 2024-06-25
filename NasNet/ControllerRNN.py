import torch
import torch.nn as nn
import torch.optim as optim
import Operations   # NASNet search space의 operation

class ControllerRNN(nn.Module):
    def __init__(self, num_layers, hidden_size, num_operations):
        super(ControllerRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_operations = num_operations
        self.rnn = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_operations)

    def forward(self, x, hx, cx):
        hx, cx = self.rnn(x, (hx, cx))
        logits = self.fc(hx)
        return logits, hx, cx

# Initialize the controller RNN
controller = ControllerRNN(num_layers=2, hidden_size=100, num_operations=len(Operations.operations))

# Example input to the controller
x = torch.zeros(1, 100)
hx, cx = torch.zeros(1, 100), torch.zeros(1, 100)
logits, hx, cx = controller(x, hx, cx)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, operation1, operation2, combine_op):
        super(Block, self).__init__()
        self.op1 = Operations.operations[operation1](in_channels, out_channels)
        self.op2 = Operations.operations[operation2](in_channels, out_channels)
        self.combine_op = combine_op

    def forward(self, x1, x2):
        y1 = self.op1(x1)
        y2 = self.op2(x2)
        if self.combine_op == 'add':
            return y1 + y2
        elif self.combine_op == 'concat':
            return torch.cat([y1, y2], dim=1)

class Cell(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(Cell, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            operation1 = 'identity'  # Randomly select or use controller RNN to select
            operation2 = '3x3 convolution'  # Randomly select or use controller RNN to select
            combine_op = 'add'  # Randomly select or use controller RNN to select
            self.blocks.append(Block(in_channels, out_channels, operation1, operation2, combine_op))

    def forward(self, x):
        states = [x]
        for block in self.blocks:
            s1 = states[-1]  # Randomly select or use controller RNN to select
            s2 = states[0]  # Randomly select or use controller RNN to select
            new_state = block(s1, s2)
            states.append(new_state)
        return torch.cat(states, dim=1)

# Example usage
cell = Cell(in_channels=32, out_channels=64, num_blocks=5)
input_tensor = torch.randn(1, 32, 32, 32)
output = cell(input_tensor)

optimizer = optim.Adam(controller.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    controller.train()
    hx, cx = torch.zeros(1, 100), torch.zeros(1, 100)
    x = torch.zeros(1, 100)
    logits, hx, cx = controller(x, hx, cx)
    loss = criterion(logits, torch.tensor([0]))  # Dummy target, replace with actual target
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')
