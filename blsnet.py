import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

# Define the neural network architecture
class BLSNet(nn.Module):
    def __init__(self):
        super(BLSNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Load a sample dataset of BLS data
df = pd.read_csv('bls_data.csv')
X = torch.tensor(df[['Unemployment Rate', 'Inflation Rate', 'Labor Force Participation Rate', 'Median Weekly Earnings']].values, dtype=torch.float32)
y = torch.tensor(df[['Job Growth']].values, dtype=torch.float32)

# Instantiate the neural network and define the loss function and optimizer
net = BLSNet()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

# Train the neural network on the BLS data
losses = []
for epoch in range(100):
    optimizer.zero_grad()
    outputs = net(X)
    loss = criterion(outputs, y)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

# Use the trained network to make predictions on new BLS data
test_X = torch.tensor([[4.0, 2.0, 63.0, 1000], [5.0, 3.0, 62.0, 900]], dtype=torch.float32)
test_y = net(test_X)
print(test_y)

# Visualize the training loss over time
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.show()