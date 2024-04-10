import torch
from tqdm import tqdm


'''
This file contains training and testing evaluations scipts to calculate accuracy and loss

'''

########################## Define helper functions to train, test and measure #####################

def GetCorrectPredCount(pPrediction, pLabels):

  '''

  This function computes the number of correct predictions by comparing
  the predicted labels (with the highest probability) against the true labels.

  '''

  return pPrediction.eq(pLabels).sum().item()



def train_ocp(model, device, train_loader, optimizer, criterion,scheduler= None, train_losses=[], train_acc=[]):

  '''

  This function trains the neural network using the training data

  '''

  model.train()  # Set the model to training mode
  pbar = tqdm(train_loader)  # Wrap the data loader with tqdm for a progress bar

  train_loss = 0  # Initialize total training loss
  correct = 0  # Initialize total number of correct predictions
  processed = 0  # Initialize total number of processed samples

  for batch_idx, (data, target) in enumerate(pbar):
    # Move the batch data and labels to the specified device (GPU)
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()  # Clear the gradients of all optimized variables
    # Forward pass: compute predicted outputs by passing inputs to the model
    pred = model(data)

    # Compute loss: calculate the batch loss by comparing predicted and true labels
    loss = criterion(pred, target)
    train_loss += loss.item()  # Aggregate the loss

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    optimizer.step()  # Perform a single optimization step (parameter update)
    scheduler.step()  # update the learning rate as in OCP, it updayed at each batch

    predicted_ = pred.argmax(dim=1)  # Get the index of the max log-probability
    correct += GetCorrectPredCount(predicted_, target)  # Update total correct predictions for the batch
    processed += len(data)  # Update total processed samples of batch

    # Update progress bar description with current loss and accuracy
    pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  # Calculate and store the average accuracy and loss for this training epoch of training data
  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))




def train(model, device, train_loader, optimizer, criterion,scheduler= None, train_losses=[], train_acc=[]):

  '''

  This function trains the neural network using the training data

  '''

  model.train()  # Set the model to training mode
  pbar = tqdm(train_loader)  # Wrap the data loader with tqdm for a progress bar

  train_loss = 0  # Initialize total training loss
  correct = 0  # Initialize total number of correct predictions
  processed = 0  # Initialize total number of processed samples

  for batch_idx, (data, target) in enumerate(pbar):
    # Move the batch data and labels to the specified device (GPU)
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()  # Clear the gradients of all optimized variables
    # Forward pass: compute predicted outputs by passing inputs to the model
    pred = model(data)

    # Compute loss: calculate the batch loss by comparing predicted and true labels
    loss = criterion(pred, target)
    train_loss += loss.item()  # Aggregate the loss

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    optimizer.step()  # Perform a single optimization step (parameter update)

    predicted_ = pred.argmax(dim=1)  # Get the index of the max log-probability
    correct += GetCorrectPredCount(predicted_, target)  # Update total correct predictions for the batch
    processed += len(data)  # Update total processed samples of batch

    # Update progress bar description with current loss and accuracy
    pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  # Calculate and store the average accuracy and loss for this training epoch of training data
  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))




def test(model, device, test_loader, criterion, test_losses=[], test_acc=[]):

  '''

  This function sets the neural network in testing mode for inference

  '''

  model.eval()  # Set the model to evaluation mode
  test_loss = 0  # Initialize total test loss
  correct = 0  # Initialize total number of correct predictions
  misclassified_samples = []

  with torch.no_grad():  # Disable gradient calculation
    for batch_idx, (data, target) in enumerate(test_loader):
      # Move the batch data and labels to the specified device (GPU)
      data, target = data.to(device), target.to(device)

      output = model(data)  # Compute output by passing inputs to the model
      test_loss += criterion(output, target).item()  # Sum up batch loss

      predicted_ = output.argmax(dim=1)  # Get the index of the max log-probability
      correct += GetCorrectPredCount(predicted_, target)  # Update total correct predictions for each batch in test data

      if len(misclassified_samples)<10:
         for i in range(len(target)):
            if predicted_[i] != target[i]:
               misclassified_samples.append((data[i].cpu(), target[i].cpu(), predicted_[i].cpu()))
               if len(misclassified_samples) == 10:
                break

  # Calculate and store the average loss and accuracy for this test run
  test_loss /= len(test_loader.dataset)
  test_acc.append(100. * correct / len(test_loader.dataset))
  test_losses.append(test_loss)

  # Print test results
  print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))
  return misclassified_samples
