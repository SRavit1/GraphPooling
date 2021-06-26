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

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv4(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv_final(x, edge_index)
        x = torch.nn.functional.relu(x)

        x = torch_geometric.nn.global_mean_pool(x, batch)

        x = torch.nn.functional.log_softmax(x, dim=1)

        return x

class GCN_model(torch.nn.Module):
    pool_out_indices_default = {"x":0, "edge_index":1, "batch":3}
    def __init__(self, num_node_features, num_classes, pool_layer=None, 
            pool_out_indices=pool_out_indices_default, pool_ratio=0.85):
        super(GCN_model, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(num_node_features, hiddenSize)
        self.conv2 = torch_geometric.nn.GCNConv(hiddenSize, hiddenSize)
        self.conv3 = torch_geometric.nn.GCNConv(hiddenSize, hiddenSize)
        self.conv4 = torch_geometric.nn.GCNConv(hiddenSize, hiddenSize)
        self.conv_final = torch_geometric.nn.GCNConv(hiddenSize, num_classes)

        self.pool_layer = pool_layer
        if pool_layer:
            if pool_layer == torch_geometric.nn.EdgePooling:
                self.pool1 = self.pool_layer(hiddenSize)
                self.pool2 = self.pool_layer(hiddenSize)
            else:
                self.pool1 = self.pool_layer(hiddenSize, pool_ratio)
                self.pool2 = self.pool_layer(hiddenSize, pool_ratio)
        self.pool_out_indices = pool_out_indices

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        #x = torch.nn.functional.dropout(x, training=self.training)

        if self.pool_layer:
            pool_out = self.pool1(x, edge_index, batch=batch)
            x = pool_out[self.pool_out_indices["x"]]
            edge_index = pool_out[self.pool_out_indices["edge_index"]]
            batch = pool_out[self.pool_out_indices["batch"]]

        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.nn.functional.relu(x)

        if self.pool_layer:
            pool_out = self.pool2(x, edge_index, batch=batch)
            x = pool_out[self.pool_out_indices["x"]]
            edge_index = pool_out[self.pool_out_indices["edge_index"]]
            batch = pool_out[self.pool_out_indices["batch"]]

        x = self.conv4(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv_final(x, edge_index)
        x = torch.nn.functional.relu(x)

        x = torch_geometric.nn.global_mean_pool(x, batch)

        x = torch.nn.functional.log_softmax(x, dim=1)

        return x

class GCN(GCN_model):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__(num_node_features, num_classes)

class GCN_SAGPool(GCN_model):
    def __init__(self, num_node_features, num_classes):
        super(GCN_SAGPool, self).__init__(num_node_features, num_classes, 
            pool_layer=torch_geometric.nn.SAGPooling)

class GCN_TopKPool(GCN_model):
    def __init__(self, num_node_features, num_classes):
        super(GCN_TopKPool, self).__init__(num_node_features, num_classes, 
            pool_layer=torch_geometric.nn.TopKPooling)

class GCN_EdgePool(GCN_model):
    def __init__(self, num_node_features, num_classes):
        pool_out_indices = {"x":0, "edge_index":1, "batch":2}
        super(GCN_EdgePool, self).__init__(num_node_features, num_classes, 
            pool_layer=torch_geometric.nn.EdgePooling, 
            pool_out_indices=pool_out_indices)