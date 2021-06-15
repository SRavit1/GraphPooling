import torch
import torch_geometric
import models
from modelUtils import train, validate, load_dataset, visualize_training

# Load dataset
enzymes_dataset, enzymes_train_loader, enzymes_test_loader = load_dataset('ENZYMES', train_fraction=0.6, batch_size=1)
print("[INFO] Loaded dataset.")

# Train
epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.GCN(enzymes_dataset.num_node_features, enzymes_dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_function = torch.nn.functional.nll_loss

logs = train(model, epochs, enzymes_train_loader, enzymes_test_loader, optimizer, loss_function, validate_frequency=2)
print("[INFO] Trained model.")

visualize_training(logs)