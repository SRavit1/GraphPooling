import torch
import torch_geometric
from matplotlib import pyplot as plt

def load_dataset(datasetName, train_fraction=0.9, batch_size=32):
  dataset = torch_geometric.datasets.TUDataset(root='/tmp/' + datasetName, name=datasetName)
  dataset = dataset.shuffle()

  cutoff = int(len(dataset)*train_fraction)
  dataset_train = dataset[:cutoff]
  dataset_test = dataset[cutoff:]

  train_loader = torch_geometric.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
  test_loader = torch_geometric.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
  return dataset, train_loader, test_loader

def train(model, epochs, train_data_loader, test_data_loader, optimizer, loss_function, validate_frequency=5):
  logs = {"accuracy":{}, "loss":{}}
  model.train()
  for epoch in range(epochs):
    if (epoch%validate_frequency == 0):
      print("Epoch {} of {}".format(epoch, epochs), end='\t')
      accuracy, loss = validate(model, test_data_loader, loss_function)
      
      logs["accuracy"][epoch] = accuracy
      logs["loss"][epoch] = loss
      
      model.train()
    for batch in train_data_loader:
      optimizer.zero_grad()
      loss = loss_function(model(batch), batch.y)
      loss.backward()
      optimizer.step()
  return logs

def validate(model, test_data_loader, loss_function):
  model.eval()
  correct = 0
  total = 0
  totalLoss = 0
  for batch in test_data_loader:
    prediction = model(batch)

    loss = loss_function(prediction, batch.y)
    totalLoss += loss
    
    if (torch.argmax(prediction) == batch.y[0]):
      correct += 1
    total += 1

  accuracy = correct/total
  loss = totalLoss/total
  print("Accuracy: {:.4f}\tLoss: {:.4f}".format(accuracy, loss))
  return accuracy, loss

def visualize_training(logs):
  fig, ax1 = plt.subplots()
  ax1.plot(list(logs["accuracy"].keys()), list(logs["accuracy"].values()), label="accuracy", color="blue")
  ax1.set_xlabel('epochs')
  ax2 = ax1.twinx()
  ax2.plot(list(logs["loss"].keys()), list(logs["loss"].values()), label="loss", color="orange")
  plt.title("Visualization of accuracy and loss during training.")
  ax1.legend()
  ax2.legend()

  fig.tight_layout()
  plt.show()