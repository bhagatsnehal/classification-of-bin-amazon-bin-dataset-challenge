#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import os
import sys
import logging

from torchvision.datasets import ImageFolder

import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    print("START TESTING")
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data=data.to(device)
            target=target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Accuracy: {:.0f}% ({}/{})\n".format(
            100.0 * correct / len(test_loader.dataset), correct, len(test_loader.dataset)
        )
    )
       
    pass


def train(model, train_loader, criterion, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    model.train()
    for e in range(epochs):
        print("START TRAINING")
        running_loss=0
        correct=0
        for data, target in train_loader:
            data=data.to(device)
            target=target.to(device)
            optimizer.zero_grad()
            pred = model(data)             #No need to reshape data since CNNs take image inputs
            loss = criterion(pred, target)
            running_loss+=loss.item() * data.size(0)
            loss.backward()
            optimizer.step()
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, \
             Accuracy {100*(correct/len(train_loader.dataset))}%")
    
    pass
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    
    model = models.resnet50(pretrained = True)
    
#   Unfreeze(grad = True) or Freeze(grad = False) all parameters of the pretrained part of the loaded model. 
#   This does not affect the parameter of the fully connected layer that's defined after
    for param in model.parameters():
        param.requires_grad = True
        
    num_features = model.fc.in_features # 2048 for resnet50, 512 for resnet18
    
    model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128,5)
    )
   
    return model
    pass

def create_data_loaders(traindir, testdir, batch_size, test_batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    training_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testing_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = ImageFolder(traindir, transform=training_transform)
    testset = ImageFolder(testdir, transform=testing_transform)

    logger.info("Batch Size {}".format(args.batch_size))
    logger.info("Batch Size {}".format(args.test_batch_size))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, test_batch_size, shuffle=True)

    return train_loader, test_loader
  
    pass

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    model.to(device)
    
    # Create train and test loaders
    train_loader, test_loader =  create_data_loaders(args.train_data_dir, args.test_data_dir, args.batch_size, args.test_batch_size)
    
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train(model, train_loader, loss_criterion, optimizer, args.epochs, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    print(f"Model Save Directory is {args.model_dir}")

    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size",type = int,default = 64, help = "input batch size for training(default: 64)"
    )
    parser.add_argument(
        "--test-batch-size",type = int,default = 10, help = "input batch size for testing(default: 10)"
    )
    parser.add_argument(
        "--epochs",type = int,default = 4, help = "number of epochs(default: 4)"
    )
    parser.add_argument(
        "--lr", type = float, default = 1.0, metavar = "LR",  help = "learning rate(default: 1.0)"
    )
    parser.add_argument(
        "--train-data-dir", type = str, default = os.environ["SM_CHANNEL_TRAINING"]
    )
    parser.add_argument(
        "--test-data-dir", type = str, default = os.environ["SM_CHANNEL_TEST"]
    )
    parser.add_argument(
        "--model-dir", type = str, default = os.environ["SM_MODEL_DIR"]
    )
    
    
    args=parser.parse_args()
    
    main(args)
