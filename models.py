import torch
import torch_geometric

hiddenSize = 6

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

class GCN_model(torch.nn.Module):
    pool_out_indices_default = {"x":0, "edge_index":1, "batch":3}
    def __init__(self, num_node_features, num_classes, pool_layers=None, pool_out_indices=pool_out_indices_default):
        super(GCN_model, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(num_node_features, hiddenSize)
        self.conv2 = torch_geometric.nn.GCNConv(hiddenSize, num_classes)

        self.pool_layers = pool_layers
        self.pool_out_indices = pool_out_indices

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)

        if self.pool_layers and len(self.pool_layers) >= 1:
            pool_out = self.pool_layers[0](x, edge_index, batch=batch)
            x = pool_out[self.pool_out_indices["x"]]
            edge_index = pool_out[self.pool_out_indices["edge_index"]]
            batch = pool_out[self.pool_out_indices["batch"]]

        x = self.conv2(x, edge_index)

        if self.pool_layers and len(self.pool_layers) >= 2:
            pool_out = self.pool_layers[1](x, edge_index, batch=batch)
            x = pool_out[self.pool_out_indices["x"]]
            edge_index = pool_out[self.pool_out_indices["edge_index"]]
            batch = pool_out[self.pool_out_indices["batch"]]

        x = torch_geometric.nn.global_mean_pool(x, batch)

        x = torch.nn.functional.log_softmax(x, dim=1)

        return x

class GCN(GCN_model):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__(num_node_features, num_classes)

class GCN_SAGPool(GCN_model):
    def __init__(self, num_node_features, num_classes):
        pool1 = torch_geometric.nn.SAGPooling(hiddenSize, 0.25)
        pool2 = torch_geometric.nn.SAGPooling(num_classes, 0.25)
        pool_layers = [pool1, pool2]
        super(GCN_SAGPool, self).__init__(num_node_features, num_classes, pool_layers=pool_layers)

class GCN_TopKPool(GCN_model):
    def __init__(self, num_node_features, num_classes):
        pool1 = torch_geometric.nn.TopKPooling(hiddenSize, 0.25)
        pool2 = torch_geometric.nn.TopKPooling(num_classes, 0.25)
        pool_layers = [pool1, pool2]
        super(GCN_TopKPool, self).__init__(num_node_features, num_classes, pool_layers=pool_layers)

class GCN_EdgePool(GCN_model):
    def __init__(self, num_node_features, num_classes):
        pool1 = torch_geometric.nn.EdgePooling(hiddenSize)
        pool2 = torch_geometric.nn.EdgePooling(num_classes)
        pool_layers = [pool1, pool2]
        pool_out_indices = {"x":0, "edge_index":1, "batch":2}
        super(GCN_EdgePool, self).__init__(num_node_features, num_classes, pool_layers=pool_layers, 
            pool_out_indices=pool_out_indices)