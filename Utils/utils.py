import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import v2
from torch_lr_finder import LRFinder
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

################### set device ########################



def setdevice():
  # check if CUDA is available or not
  cuda = torch.cuda.is_available()
  # print("CUDA Available?", cuda)
  device = torch.device("cuda" if cuda else "cpu")       #setting the device on which computations will be executed
  print("Device set to: ",device)
  return device



############################ Get optimizer #############################

def get_optimizer( model, optname="SGD", learningrate = 0.01,weightdecay = 0,momentum = 0.9):

  '''

  Get optimizer object of required algorith with learning rate and momentum. default optimizer = "SGD"
  # Available values for <optname> SGD, ADAM

  # Momentum is a technique used to prevent GD from getting stuck in local minima
  Example of how it works internally:
    velocity = momentum * velocity - learning_rate * gradient
    parameters = parameters + velocity


  '''

  if optname == "SGD":
    print("Optimizer has been set to SGD...")
    return optim.SGD(model.parameters(), lr = learningrate, momentum=momentum)
  elif optname == "ADAM":
    print("Optimizer has been set to ADAM...")
    return optim.Adam(model.parameters(), lr = learningrate, weight_decay = weightdecay)
  else:
    raise ValueError('Incorrect optimizer algo string...pass valid name from available values')



############################ Get scheduler #############################


def get_scheduler(optimizerval,onecycle = True, **kwargs):

  '''

  onecycle == False >initialize scheduler for learning rate to be slower after n steps i.e after stepsize; learning rate updates to gamma*learningrate
  onecycle == True > initiaze pytorch onecycle policy. An assesment via LR Finder library in necessary to use this effectively

  '''

  if onecycle == False:
    print("Default Scheduler Activated...")
    return optim.lr_scheduler.StepLR(optimizerval, **kwargs)
  elif onecycle == True:
    print("One Cycle Policy Activated...")
    return optim.lr_scheduler.OneCycleLR(optimizer=optimizerval,**kwargs)
  else:
    raise ValueError('Incorrect Boolean, takes either True or False to activate one cycle policy')




############################ Get loss function #########################

def get_loss(loss_name="nll_loss"):

  '''

  get loss function object of specified loss criteria, default value = "nll_loss"

  '''

  if loss_name=="nll_loss":
    print("Loss evaluated through Negative Log Likelihood...")
    return F.nll_loss
  elif loss_name == 'crossentropy':
    print("Loss evaluated through Cross Entropy...")
    return F.cross_entropy
  else:
    raise ValueError('Incorrect loss name string...pass valid name from available values')





######## Get image from tensor ##############


def get_image_from_tensor(img_tensor,mean_list, std_list):
    to_pil = v2.ToPILImage()

    # Unnormalize the image tensor
    for t, m, s in zip(img_tensor, mean_list, std_list):
        t.mul_(s).add_(m)

    # Convert the tensor to a PIL Image
    img = to_pil(img_tensor)

    return img







################## Display Samples ###########################

def post_display(train_loader,label_map,mean_list, std_list):

  '''

  Display some of the samples from training data

  '''

  # Get a batch of data and labels from the training DataLoader
  batch_data, batch_label = next(iter(train_loader))

    # Create a new figure for plotting
  fig, axes = plt.subplots(4, 4, figsize=(8, 8))



  # Loop through 16 samples in the batch
  for i in range(16):
      # Display the image (convert from tensor to numpy array) in RGB

      img_tensor = batch_data[i]

      img = get_image_from_tensor(img_tensor,mean_list, std_list)

      # Convert the PIL Image to a numpy array and display it
      image = np.array(img)

      axes[i // 4, i % 4].imshow(image)

      # Set the title of the subplot to the corresponding label
      axes[i // 4, i % 4].set_title(label_map[batch_label[i].item()] + f", Label: {batch_label[i].item()}")

      # Remove x and y ticks for cleaner visualization
      axes[i // 4, i % 4].axis("off")

  # Ensure tight layout for better visualization
  plt.tight_layout()

  # Show the entire figure with subplots
  plt.show()




################## Display Accuract & Loss Plots ###########################

def post_accuracyplots(train_losses,test_losses,train_acc,test_acc):

  '''

  Plot Accuracy and Loss plots on training and testing

  '''

  fig, axs = plt.subplots(2,2,figsize=(15,10))

    # Set the title for the entire figure
  fig.suptitle(f"Plots", fontsize=16)

  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")




##################### LR finder ##########################


def initiateLRfinder(train_loader, model, optimizer, criterion, device ='cpu'):

  '''
  To find the Max LR of One Cycle Polic
  '''

  lr_finder = LRFinder(model, optimizer, criterion, device)
  lr_finder.range_test(train_loader, end_lr=1, num_iter=200, step_mode="exp")
  lr_finder.plot()
  return lr_finder



################# Get LR at each epoch #################

def epochLR(epoch, scheduler,lr_epoch):

  '''
  Get updated LR at each epoch
  '''

  current_lr = scheduler.get_last_lr()  # Get the latest learning rate
  lr_epoch.append(round(current_lr[0],4))
  print(f'Current Epoch: {epoch} // Learning rate achieved at last epoch: {[round(current_lr[0],4)]}')
  return lr_epoch



###############################

def showmisclassifiedsamples(misclassified_samples,label_map,plottitle,mean_list, std_list):

  '''

  Display some of the samples from training data

  '''

    # Create a new figure for plotting
  fig, axes = plt.subplots(2, 5, figsize=(12, 7))

  # Set the title for the entire figure
  fig.suptitle(f"Misclassified Images - {plottitle}", fontsize=16)

  # Loop through 16 samples in the batch
  for i in range(len(misclassified_samples)):
      one_sample = misclassified_samples[i]
      # Display the image (convert from tensor to numpy array) in RGB

      img_tensor = one_sample[0].clone().detach()
      img = get_image_from_tensor(img_tensor,mean_list,std_list)

      # Convert the PIL Image to a numpy array and display it
      image = np.array(img)

      axes[i // 5, i % 5].imshow(image)

      # Set the title of the subplot to the corresponding label
      axes[i // 5, i % 5].set_title(f"Actual: {label_map[one_sample[1].item()]}"+" , "+f"Predicted: {label_map[one_sample[2].item()]}", fontsize=8)

      # Remove x and y ticks for cleaner visualization
      axes[i // 5, i % 5].axis("off")

  # Ensure tight layout for better visualization
  plt.tight_layout()

  # Show the entire figure with subplots
  plt.show()



################## list folder structure #################

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        if '.git' in root or 'pycache' in root.lower():
            continue
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
          print('{}{}'.format(subindent, f))





######################################## GRAD CAM CODE ####################################






def showgramcam(model,misclassified_samples,label_map,plottitle,mean_list, std_list,device):

  '''

  Display gradcam on images

  '''

  #Get target layer to pass to gradcam
  target_layers = [model.layer3[-1]]

  # Create a new figure for plotting
  fig, axes = plt.subplots(4, 5, figsize=(22, 25))

  # Set the title for the entire figure
  fig.suptitle(f"GRADCAM Images - {plottitle}", fontsize=16)


  # Loop through 10 samples (assuming there are exactly 10 as stated)
  for i, sample in enumerate(misclassified_samples):
      img_tensor, actual_label, predicted_label = sample[0], sample[1], sample[2]

      # Prepare the image tensor for visualization and GradCAM processing
      img_tensor = img_tensor.clone().detach()
      input_tensor = img_tensor.unsqueeze(0).to(device)

      # Convert tensor to PIL for consistent visualization handling
      pil_img = get_image_from_tensor(img_tensor.cpu(),mean_list, std_list)

      # Convert the PIL Image to a numpy array and normalize
      image = np.array(pil_img).astype(np.float32) / 255

      # Define targets for GradCAM
      actual_target = [ClassifierOutputTarget(actual_label.item())]
      predicted_target = [ClassifierOutputTarget(predicted_label.item())]

      # Generate GradCAM results
      grayscale_cam_actual = cam(input_tensor=input_tensor, targets=actual_target)[0, :]
      grayscale_cam_pred = cam(input_tensor=input_tensor, targets=predicted_target)[0, :]

      # Visualizations for actual and predicted
      visualization_actual = show_cam_on_image(image, grayscale_cam_actual, use_rgb=True)
      visualization_pred = show_cam_on_image(image, grayscale_cam_pred, use_rgb=True)

      # Plot actual visualization
      ax_actual = axes[2 * (i // 5) , i % 5]
      ax_actual.imshow(visualization_actual)
      ax_actual.set_title(f"Actual: {label_map[actual_label.item()]}"+" , "+f"Predicted: {label_map[predicted_label.item()]}", fontsize=10)
      ax_actual.set_xlabel(f"Image {i} ; <GradCAM on Actual>", fontsize=10)
      # ax_actual.axis("off")

      # Plot predicted visualization
      ax_pred = axes[2 * (i // 5) + 1, i % 5]
      ax_pred.imshow(visualization_pred)
      ax_pred.set_title(f"Actual: {label_map[actual_label.item()]}"+" , "+f"Predicted: {label_map[predicted_label.item()]}", fontsize=10)
      ax_pred.set_xlabel(f"Image {i} ; <GradCAM on Predicted>", fontsize=10)
      # ax_pred.axis("off")


  # Ensure tight layout for better visualization
  plt.tight_layout()

  # Show the entire figure with subplots
  plt.show()