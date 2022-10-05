#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import os
import argparse


#TODO: Import dependencies for Debugging andd Profiling


   


def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    #model.to(device).eval()
    return model

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 2))
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_dataset_path = os.path.join(data, "training_set")
    test_dataset_path = os.path.join(data, "test_set")

    training_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    
    testing_transform = transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        transforms.ToTensor()])

    

    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=training_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=testing_transform)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_data_loader, test_data_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model=net()
    model=model.to(device)

    train_loader, test_loader = create_data_loaders(args.data, int(args.batch_size))
    
    
    
    #test(model, test_loader,loss_criterion, device, hook)
    
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, test_loader, args.epochs, loss_criterion, optimizer, hook)

    '''
    TODO: Test the model to see its accuracy
    '''    
    '''
    TODO: Save the trained model
    '''
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
 
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 2)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)"
    )
  
    parser.add_argument('--data', type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args=parser.parse_args()

    
    main(args)