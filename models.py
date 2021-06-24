import torch
import torch_geometric

hiddenSize = 6

# Define model
class GCN_experimental(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN_experimental, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(num_node_features, hiddenSize)
        self.conv2 = torch_geometric.nn.GCNConv(hiddenSize, hiddenSize)
        self.conv3 = torch_geometric.nn.GCNConv(hiddenSize, hiddenSize)
        self.conv4 = torch_geometric.nn.GCNConv(hiddenSize, hiddenSize)
        self.conv_final = torch_geometric.nn.GCNConv(hiddenSize, num_classes)

        self.pool1 = torch_geometric.nn.SAGPooling(hiddenSize, 0.75)
        self.pool2 = torch_geometric.nn.SAGPooling(num_classes, 0.75)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        
        #x = torch.nn.functional.dropout(x, training=self.training)
        #x, edge_index, edge_attr, batch, perm, scores = self.pool1(x, edge_index, batch=batch)
        
        x = self.conv3(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv4(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv_final(x, edge_index)
        x = torch.nn.functional.relu(x)

        #x = torch.nn.functional.dropout(x, training=self.training)
        #x, edge_index, edge_attr, batch, perm, scores = self.pool2(x, edge_index, batch=batch)

        x = torch_geometric.nn.global_mean_pool(x, batch)

        x = torch.nn.functional.log_softmax(x, dim=1)

        return x

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
        
        x = torch_geometric.nn.global_mean_pool(x, data.batch)
        x = torch.nn.functional.log_softmax(x, dim=1)

        return x



class GCN_SAGPool(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN_SAGPool, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(num_node_features, hiddenSize)
        self.conv2 = torch_geometric.nn.GCNConv(hiddenSize, num_classes)
        self.pool1 = torch_geometric.nn.SAGPooling(hiddenSize, 0.25)
        self.pool2 = torch_geometric.nn.SAGPooling(num_classes, 0.25)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)

        x, edge_index, edge_attr, batch, perm, scores = self.pool1(x, edge_index, batch=batch)

        x = self.conv2(x, edge_index)

        x, edge_index, edge_attr, batch, perm, scores = self.pool2(x, edge_index, batch=batch)
        x = torch_geometric.nn.global_mean_pool(x, batch)


        x = torch.nn.functional.log_softmax(x, dim=1)

        return x

class GCN_TopKPool(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN_TopKPool, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(num_node_features, hiddenSize)
        self.conv2 = torch_geometric.nn.GCNConv(hiddenSize, num_classes)
        self.pool1 = torch_geometric.nn.TopKPooling(hiddenSize, 0.25)
        self.pool2 = torch_geometric.nn.TopKPooling(num_classes, 0.25)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)

        x, edge_index, edge_attr, batch, perm, scores = self.pool1(x, edge_index, batch=batch)

        x = self.conv2(x, edge_index)

        x, edge_index, edge_attr, batch, perm, scores = self.pool2(x, edge_index, batch=batch)
        x = torch_geometric.nn.global_mean_pool(x, batch)


        x = torch.nn.functional.log_softmax(x, dim=1)

        return x

class GCN_EdgePool(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN_EdgePool, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(num_node_features, hiddenSize)
        self.conv2 = torch_geometric.nn.GCNConv(hiddenSize, num_classes)
        self.pool1 = torch_geometric.nn.EdgePooling(hiddenSize)
        self.pool2 = torch_geometric.nn.EdgePooling(num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)

        x, edge_index, batch, _ = self.pool1(x, edge_index, batch=batch)

        x = self.conv2(x, edge_index)

        x, edge_index, batch, _ = self.pool2(x, edge_index, batch=batch)
        x = torch_geometric.nn.global_mean_pool(x, batch)


        x = torch.nn.functional.log_softmax(x, dim=1)

        return x