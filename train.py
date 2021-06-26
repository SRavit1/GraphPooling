import torch
import torch_geometric
import models
from modelUtils import train, validate, load_dataset
from trainingUtils import visualize_training, save_logs
import numpy as np
import os

torch.manual_seed(0)
np.random.seed(seed=0)

# Load dataset
dataset_train = torch_geometric.datasets.GNNBenchmarkDataset(root='/tmp/' + 'MNIST', name='MNIST', split='train')
dataset_test = torch_geometric.datasets.GNNBenchmarkDataset(root='/tmp/' + 'MNIST', name='MNIST', split='test')

dataset_train = dataset_train.shuffle()
dataset_test = dataset_test.shuffle()

train_idx = np.random.randint(len(dataset_train), size=1500)
test_idx = np.random.randint(len(dataset_test), size=500)
dataset_train = dataset_train[train_idx]
dataset_test = dataset_test[test_idx]
"""

dataset = torch_geometric.datasets.TUDataset(root='/tmp/' + 'ENZYMES', name='ENZYMES')
dataset = dataset.shuffle()

cutoff = int(len(dataset)*0.8)
dataset_train = dataset[:cutoff]
dataset_test = dataset[cutoff:]
"""

train_loader, test_loader = load_dataset(dataset_train, dataset_test, batch_size=128)

print("[INFO] Loaded dataset.")

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 40
#model_classes = {"GCN": models.GCN, "GCN SAGPool": models.GCN_SAGPool, "GCN TopKPool": models.GCN_TopKPool, "GCN EdgePool": models.GCN_EdgePool}
model_classes = {"GCN Experimental": models.GCN_experimental, "GCN": models.GCN}

all_training_logs = {}
for (model_name, model_class) in model_classes.items():
  print("[INFO] Begin training " + model_name + ".")

  model = model_class(dataset_train.num_node_features, dataset_train.num_classes).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
  loss_function = torch.nn.functional.nll_loss

  training_logs = train(model, epochs, train_loader, test_loader, optimizer, loss_function, validate_frequency=1)
  all_training_logs[model_name] = training_logs

  print("[INFO] Finished training " + model_name + ".")

  model_save_path = os.path.join(os.getcwd(), "trainedModels", model_name+".torch")
  torch.save(model.state_dict(), model_save_path)

  print("[INFO] Saved trained", model_name, "model to path", model_save_path)

save_logs(all_training_logs, "trainingLogs.json")
visualize_training(all_training_logs)