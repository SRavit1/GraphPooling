import torch
import torch_geometric

hiddenSize = 16

# Define model
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(num_node_features, hiddenSize)
        self.conv2 = torch_geometric.nn.GCNConv(hiddenSize, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        
        x = torch.mean(x, 0)
        x = torch.unsqueeze(x, 0)

        x = torch.nn.functional.log_softmax(x, dim=1)

        return x