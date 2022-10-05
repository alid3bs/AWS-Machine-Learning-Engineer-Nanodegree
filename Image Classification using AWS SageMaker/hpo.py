#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import smdebug.pytorch as smd
import argparse
import os

#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader,criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    los, acc = 0, 0
    for inputs, datalabels in test_loader:
        inputs=inputs.to(device)
        datalabels=datalabels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, datalabels)
        _, predictions = torch.max(outputs, 1)
        los += loss.item() * inputs.size(0)
        acc += torch.sum(predictions == datalabels.data)

    total_loss = los // len(test_loader)
    total_acc = acc // len(test_loader)
    print(f'Test set: Accuracy: {acc}/{len(test_loader.dataset)} = {100*total_acc}%),\t Testing Loss: {total_loss}')
    
def train(model, train_loader, criterion, optimizer, epochs,device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    image_dataset={'train':train_loader}
    
    for epoch in range(epochs):
        
            model.train()
            los = 0
            acc = 0

            for inputs, datalabels in image_dataset["train"]:
                inputs=inputs.to(device)
                datalabels=datalabels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, datalabels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, prediction = torch.max(outputs, 1)
                los += loss.item() * inputs.size(0)
                acc += torch.sum(prediction == datalabels.data)
            epoch_loss = los // len(image_dataset["train"])
            epoch_acc = acc // len(image_dataset["train"])
            print("Epoch {}, loss: {}, acc: {}\n".format(epoch, epoch_loss, epoch_acc))           
    return model
    
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
    #data_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle=True)
    #return data_loader


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
    device =   torch.device("cpu")
    model=net()
    model=model.to(device)
    


    train_loader, test_loader = create_data_loaders(args.data, int(args.batch_size))
    
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, loss_criterion, optimizer, args.epochs, device)
    '''
    
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader,  loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    EDIT: Specify any training args that you might need
    '''
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training",
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.01, 
        metavar="LR", 
        help="learning rate"
    )
   
    
    parser.add_argument("--model-dir", type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args=parser.parse_args()

    main(args)
