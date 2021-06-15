import torch
import torch_geometric
import models
from modelUtils import train, validate, load_dataset, visualize_training, save_logs

# Load dataset
enzymes_dataset, enzymes_train_loader, enzymes_test_loader = load_dataset('ENZYMES', train_fraction=0.6, batch_size=32)
print("[INFO] Loaded dataset.")

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 100
model_classes = {"GCN": models.GCN, "GCN SAGPool": models.GCN_SAGPool, "GCN TopKPool": models.GCN_TopKPool}

all_training_logs = {}
for (model_name, model_class) in model_classes.items():
  print("[INFO] Begin training " + model_name + ".")

  model = model_class(enzymes_dataset.num_node_features, enzymes_dataset.num_classes).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
  loss_function = torch.nn.functional.nll_loss

  training_logs = train(model, epochs, enzymes_train_loader, enzymes_test_loader, optimizer, loss_function, validate_frequency=5)
  all_training_logs[model_name] = training_logs

  print("[INFO] Finished training " + model_name + ".")

  #visualize_training(training_logs)

save_logs(all_training_logs, "trainingLogs.json")