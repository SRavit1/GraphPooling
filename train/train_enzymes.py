import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Define model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(enzymes_dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, enzymes_dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.mean(x, 0)
        x = torch.unsqueeze(x, 0)

        x = F.log_softmax(x, dim=1)

        return x

# Load dataset
enzymes_dataset = torch_geometric.datasets.TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
enzymes_dataset = enzymes_dataset.shuffle()

enzymes_train = enzymes_dataset[:540]
enzymes_test = enzymes_dataset[540:]

enzymes_train_loader = torch_geometric.data.DataLoader(enzymes_train, batch_size=1, shuffle=True)
enzymes_test_loader = torch_geometric.data.DataLoader(enzymes_test, batch_size=1, shuffle=True)

# Train
epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(epochs):
  if (epoch%5 == 0):
    print("Epoch {} of {}".format(epoch, epochs))
  for enzymes_batch in enzymes_train_loader:
    optimizer.zero_grad()
    out = model(enzymes_batch)
    loss = F.nll_loss(out, enzymes_batch.y)
    loss.backward()
    optimizer.step()

model.eval()

correct = 0
total = 0
for enzymes_batch in enzymes_test_loader:
  optimizer.zero_grad()
  out = torch.argmax(model(enzymes_batch))
  if (out == enzymes_batch.y[0]):
    correct += 1
  total += 1
print("Accuracy is {:.4f}".format(correct/total))
