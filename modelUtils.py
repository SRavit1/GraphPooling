import torch
import torch_geometric

def load_dataset(dataset_train, dataset_test, batch_size=32):
  train_loader = torch_geometric.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
  test_loader = torch_geometric.data.DataLoader(dataset_test, batch_size=batch_size)
  return train_loader, test_loader

def train(model, epochs, train_data_loader, test_data_loader, optimizer, loss_function, validate_frequency=5):
  logs = {"validation_accuracy":{}, "validation_loss":{}, "train_accuracy":{}, "train_loss":{}}
  model.train()
  for epoch in range(epochs):
    for batch in train_data_loader:
      optimizer.zero_grad()
      loss = loss_function(model(batch), batch.y)
      loss.backward()
      optimizer.step()
    if (epoch%validate_frequency == 0):
      print("Epoch {}/{}".format(epoch, epochs), end='\t')
      model.eval()

      accuracy, loss = validate(model, test_data_loader, loss_function)
      train_accuracy, train_loss = validate(model, train_data_loader, loss_function, numBatches=32)

      logs["validation_accuracy"][epoch] = accuracy
      logs["validation_loss"][epoch] = loss
      logs["train_accuracy"][epoch] = train_accuracy
      logs["train_loss"][epoch] = train_loss

      print("Valid acc: {:.4f}\tValid loss: {:.4f}\tTrain acc: {:.4f}\tTrain loss: {:.4f}".format(accuracy, loss, train_accuracy, train_loss))
      
      model.train()
  return logs

#TODO: Use Pytorch methods for accuracy/loss calculations
def validate(model, test_data_loader, loss_function, numBatches=None):
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

    if numBatches and totalBatches >= numBatches:
      break

  accuracy = float(correct/totalGraphs)
  loss = float(totalLoss/totalBatches)
  return accuracy, loss