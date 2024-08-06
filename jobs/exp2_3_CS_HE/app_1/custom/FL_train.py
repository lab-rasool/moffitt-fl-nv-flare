import os
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import KFold
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from model import ResNetLoader, NLSTDataset, NLST_CNN
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import argparse

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.app_common.app_constant import ModelName

import wandb
from nvflare.client.tracking import WandBWriter
wandb.login(key='98d00235ae0aaf85e12344ec1da6f273dfbe3fbe')
# wandb.init(
#      project="Federated_nlst", 
#      entity="Niko_k98",
#      config={"learning_rate": 0.001,"method": "Federated Learning","data": "testingset"})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

    
    
def train(net, train_loader, criterion, optimizer,device):
    print('Training')
    net.train()
    correct = 0
    total = 0   
    total_loss=0
    i=0
    for inputs, targets in train_loader:
        i+=1
        inputs, targets = inputs.to(device), targets.to(device)
        # print(inputs.shape)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
    
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        # print(" batch ", i ," of ",len(train_loader), "loss ", loss.item())
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total 
   
    return average_loss, accuracy

def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='', nargs="?")
    parser.add_argument("--val_dataset_path", type=str, default='', nargs="?")
    parser.add_argument("--batch_size", type=int, default=64, nargs="?")
    parser.add_argument("--num_workers", type=int, default=8, nargs="?")
    parser.add_argument("--local_epochs", type=int, default=1, nargs="?")
    parser.add_argument("--model_path", type=str, default="best_local_accuracy.pt", nargs="?")
    return parser.parse_args()

def main():
    # Define any data transformations
    args = define_parser()

    dataset_path = args.dataset_path.replace('\\,', ',').replace('\\ ', ' ')
    val_dataset_path = args.val_dataset_path.replace('\\,', ',').replace('\\ ', ' ')
    batch_size = args.batch_size
    num_workers = args.num_workers
    local_epochs = args.local_epochs
    model_path = args.model_path



    Mean = [0.3210, 0.3210, 0.3210]
    Std =[0.2120, 0.2120, 0.2120]

    # Mean= [0.3296, 0.3296, 0.3296]
    # Std= [0.2132, 0.2132, 0.2132]


    # Define any data transformations
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
            # transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=Mean,std=Std)
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert numpy array to tensor
        transforms.Normalize(mean=Mean,std=Std)  # Normalize to [0, 1] range
    ])

    # Create an instance of the dataset
    print(dataset_path)
    train_dataset = NLSTDataset(root_dir=dataset_path, transform=train_transforms)
    val_dataset =NLSTDataset(root_dir=val_dataset_path, transform=test_transforms)  # red
    # trainset = NLSTDataset(root_dir='/share/dept_machinelearning/Faculty/Rasool, Ghulam/Data/NLST/test', transform=train_transforms) # red


    # trainset = DICOMDataset(root_dir=r'/mnt/Dept_MachineLearning/Faculty/Rasool, Ghulam/Data/NLST/train', transform=train_transforms) # dgx
    # testset  = DICOMDataset(root_dir=r'/mnt/Dept_MachineLearning/Faculty/Rasool, Ghulam/Data/NLST/test', transform=test_transforms) # dgx

    #######################exp
    # train_ratio = 0.8
    # test_ratio = 0.2

    # # Calculate the sizes for training and testing
    # train_size = int(train_ratio * len(dataset))
    # test_size = len(dataset) - train_size

    # Split the dataset
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Apply test transforms to the test dataset
    # test_dataset.dataset.transform = test_transforms
    ###############################exp end
    # Create a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    # Example usage
    print(len(val_loader))
# exit()
# for images, labels in train_loader:
#     print(images.shape, labels)
#     break
# exit()

    lr=0.001
    net = NLST_CNN(num_classes=2)
    # torch.save(net.state_dict(), 'initial_model_weights.pt')
    # resnet_loader = (model_name='resnet18', pretrained=True, num_classes=2)
    # net = resnet_loader.get_model()
    best_accuracy=0.0
    best_global_acc=0.0
    # net=NLST_CNN()
   

    def test(input_weights ,criterion,device):
        print('Testing')
        # net = resnet_loader.get_model()

        net = NLST_CNN(num_classes=2)
        net.load_state_dict(input_weights)
        net.to(device)
        correct = 0
        total = 0   
        total_loss=0
        j=0
        # all_preds = []
        # all_targets = []
        for inputs, targets in val_loader:
            j+=1
            inputs, targets = inputs.to(device), targets.to(device)
            # print(inputs.shape)
            outputs = net(inputs)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            test_loss = total_loss / len(val_loader)
            # print(" test batch ", j ," of ",len(test_loader), "loss ", loss.item())

            with torch.no_grad():
                    _, predicted = torch.max(outputs.data, 1)
                    
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            # print('pred',predicted)
            # print('targets',targets)
            # all_preds.extend(predicted.cpu().numpy())
            # all_targets.extend(targets.cpu().numpy())

        test_accuracy = 100 * correct / total    


        return test_loss, test_accuracy

   
    test_accs=[]
    losses=[]
    test_losses=[]

    flare.init()
    # wandb.init(
    # project="Federated_nlst", 
    # entity="Niko_k98",
    # config={"learning_rate": 0.001})

    wandb_w=WandBWriter()
    
    # initial_weights_loaded = False
    global_accs=[]
    while flare.is_running():
        input_model = flare.receive()
        # torch.save(input_model.params, 'global_model_debug.pt')
        client_id = flare.get_site_name()

        if flare.is_train():
            print(f"({client_id}) current_round={input_model.current_round}, total_rounds={input_model.total_rounds}")

        
            net.load_state_dict(input_model.params)

            # #define class weights####
            # labels = torch.tensor(dataset.labels)
            # # print(type(labels))
            # class_counts = torch.bincount(labels)
            
            # class_weights = 1. / class_counts.float()
            # class_weights = class_weights.to(device)
            criterion = nn.CrossEntropyLoss()

            #####
            optimizer=optim.Adam(net.parameters(), lr=lr,weight_decay=1e-3)
            net=net.to(device)
            steps = local_epochs * len(train_loader)

            # global_accs=[]
            # test_accs=[]
            # losses=[]
            # test_losses=[]
            for epoch in range(local_epochs):
                print("="*50)
                # loss, accuracy = train(net, train_loader, criterion, optimizer,device)
                # net.train()
                correct = 0
                total = 0   
                total_loss=0
                i=0
                for inputs, targets in train_loader:
                    i+=1
                    inputs, targets = inputs.to(device), targets.to(device)
                    # print(inputs.shape)

                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    average_loss = total_loss / len(train_loader)
                    # print(" batch ", i ," of ",len(train_loader), "loss ", loss.item())
                    with torch.no_grad():
                        _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                accuracy = 100 * correct / total 
                # print(len(train_loader))
                # print(len(test_loader)
                wandb_w.log({"client_id":client_id,"training_loss":average_loss,"training_accuracy_local_epochs":accuracy})

                wandb_w.log({'Federated_round':input_model.current_round})


               
            
            print(f"({client_id}) Finished Training", "Training Loss",loss,'Training Accuracy',accuracy)
            wandb_w.log({"client_id":client_id,"training_loss":average_loss,"training_accuracy":accuracy})

            global_loss, global_accuracy = test(input_model.params,criterion,device)
            print(f"({client_id}) Evaluating received model for model selection. Accuracy on the validation set: {global_accuracy}")
            wandb_w.log({"client_id":client_id,"global_loss":global_loss,"global_accuracy":global_accuracy})
        
            
            local_loss,local_accuracy = test(net.state_dict(),criterion,device)

            print(f"({client_id}) Evaluating local trained model",'Train loss',loss, 'Test loss ',local_loss, "|, Train acc ",accuracy," Test acc ", local_accuracy)  
            wandb_w.log({"client_id":client_id,"local_loss":local_loss,"local_accuracy":local_accuracy})          
                
            if global_accuracy > best_global_acc:
                best_global_acc = global_accuracy 
                torch.save(input_model.params, 'global_model_backup.pt')


            if local_accuracy > best_accuracy:
                best_accuracy = local_accuracy
                # torch.save(net.state_dict(), model_path)
                torch.save(net.state_dict(), f'best_accuracy_client_{client_id}.pt')

           
                # print('train loss',loss, 'test loss ',test_loss, "|, train acc ",accuracy," test acc ", test_accuracy)
            global_accs.append(global_accuracy)
            
            output_model = flare.FLModel(
            params=net.cpu().state_dict(),
            metrics={"accuracy": global_accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},)
            

            # print(output_model)
            flare.send(output_model)
                

        elif flare.is_evaluate():
            accuracy = test(input_model.params)
            flare.send(flare.FLModel(metrics={"accuracy": accuracy}))

        # (7) performing submit_model task to obtain best local model
        elif flare.is_submit_model():
            model_name = input_model.meta["submit_model_name"]
            if model_name == ModelName.BEST_MODEL:
                try:
                    weights = torch.load(model_path)
                    net = net = NLST_CNN()
                    net.load_state_dict(weights)
                    flare.send(flare.FLModel(params=net.cpu().state_dict()))
                except Exception as e:
                    raise ValueError("Unable to load best model") from e
            else:
                raise ValueError(f"Unknown model_type: {model_name}")
                # torch.save(net.state_dict(), 'dl_train_weights.pt')
                # Start plotting
                
           

if __name__ == "__main__":
    main()