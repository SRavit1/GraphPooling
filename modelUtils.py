import torch
import torch_geometric

def load_dataset(dataset_train, dataset_test, batch_size=32):
  train_loader = torch_geometric.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
  test_loader = torch_geometric.data.DataLoader(dataset_test, batch_size=batch_size)
  return train_loader, test_loader

def train(model, epochs, train_data_loader, test_data_loader, optimizer, loss_function, validate_frequency=5):
  logs = {"accuracy":{}, "loss":{}}
  model.train()
  for epoch in range(epochs):
    for batch in train_data_loader:
      optimizer.zero_grad()
      loss = loss_function(model(batch), batch.y)
      loss.backward()
      optimizer.step()
    if (epoch%validate_frequency == 0):
      print("Epoch {} of {}".format(epoch, epochs), end='\t')
      model.eval()

      accuracy, loss = validate(model, test_data_loader, loss_function)
      logs["accuracy"][epoch] = accuracy
      logs["loss"][epoch] = loss
      
      model.train()
  return logs

#TODO: Use Pytorch methods for accuracy/loss calculations
def validate(model, test_data_loader, loss_function):
  correct = 0
  totalLoss = 0
  totalGraphs = 0
  totalBatches = 0
  for batch in test_data_loader:
    prediction = model(batch)

    loss = loss_function(prediction, batch.y)
    totalLoss += loss
    
    assert len(prediction) == len(batch.y), "# batches in prediction does not match # batches in target."
    for i in range(len(prediction)):
	    if (torch.argmax(prediction[i]) == batch.y[i]):
	      correct += 1
	    totalGraphs += 1
    totalBatches += 1

  accuracy = float(correct/totalGraphs)
  loss = float(totalLoss/totalBatches)
  print("Accuracy: {:.4f}\tLoss: {:.4f}".format(accuracy, loss))
  return accuracy, loss