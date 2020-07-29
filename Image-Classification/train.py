from helper import *
#Creating an object of ArgumentParser

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Directory to training images')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--arch', dest='arch', default='densenet161', action='store',choices=['vgg13', 'densenet121'], help='Architecture')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='hidden units')
parser.add_argument('--epochs', type=int, default=20, help='Epoch count')
parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU for training')
parser.set_defaults(gpu=False)

args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
device = args.gpu


#Defining the number of units in output layer
output_units = 102
    
if(data_dir == None):
    print("Data directory cannot be none")
    exit()

if(arch == None):
    arch = "densenet121"

if(arch == "densenet121"):
    input_units = 1024
 
if(arch == "vgg13"):
    input_units = 25088

if(arch!='vgg13' and arch!='densenet121'):
    print("Select either vgg13 or densenet121 model")
    exit()

if(save_dir == None):
   save_dir = "checkpoint.pth"
                    
if(learning_rate == None):
    learning_rate = 0.001
else:
    learning_rate = float(learning_rate)

if(epochs == None):
    epochs = 10
else:
    epochs = int(epochs)
                    
if(hidden_units == None):
    if(arch =="densenet121"):
        hidden_units = 500   
    elif(arch == "vgg13"):
        hidden_units = 4096
        
else:
    hidden_units = int(hidden_units)

if(device == False):
    device = "cpu"
else:
    if(torch.cuda.is_available()):
        device = "cuda"
    else:
        Print("Torch Cuda is not available!! Hence Using CPU")
        device = "cpu"
            
#print(data_dir)
#print(save_dir)
#print(hidden_units)
#print(epochs)
#print(learning_rate)
#print(device)



#Build and train the network
if(arch=='vgg13'):
    model = models.vgg13(pretrained = True)
else:
    model = models.densenet121(pretrained = True)
                    
for param in model.parameters():
    param.require_grad = False

classifier = nn.Sequential(nn.Linear(input_units,hidden_units),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                          nn.Linear(hidden_units,output_units),
                          nn.LogSoftmax(dim = 1))

model.classifier = classifier

#Define the Criterion
criterion = nn.NLLLoss()
#Define the Optimizer
optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
model.to(device)
                    
print("Started Training the Network")
steps = 0
print_every = 5
running_loss = 0
for epoch in range(epochs):
    for inputs,labels in trainloader:
        steps += 1
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        
        #Do a Forward Pass
        logps = model.forward(inputs)
        #Compute the loss using criterion
        loss = criterion(logps,labels)
        #Do a backward pass
        loss.backward()
        #Update the weights using the optimizer
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            #Now do validation here after every 5 steps
            valid_loss = 0 
            valid_accuracy = 0
            #change the model state to evaluation
            model.eval()
            with torch.no_grad():
                for inputs,labels in validloader:
                    inputs,labels = inputs.to(device),labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps,labels)
                    valid_loss += batch_loss.item()
                    #Compute the Accuracy
                    ps = torch.exp(logps)
                    top_p,top_class = ps.topk(1,dim = 1)
                    equals = top_class == labels.view(*top_p.shape)
                    valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {valid_accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train() 
print("Network has been trained successfully")
#Doing the Inference on Test Data Set
def check_accuracy_on_test_data(testloader):
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        for inputs,labels in testloader:
            inputs,labels = inputs.to(device),labels.to(device)
            logps = model.forward(inputs)
            loss = criterion(logps,labels)
            test_loss += loss.item()
            ps = torch.exp(logps)
            top_p,top_class = ps.topk(1,dim = 1)
            equals = top_class == labels.view(*top_p.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {test_accuracy/len(testloader):.3f}")

check_accuracy_on_test_data(dataloaders['testloader'])
#Save the Checkpoint for the future
model.class_to_idx = image_datasets['train_data'].class_to_idx
model.cpu()
checkpoint = {'transfer_model': arch,
              'features': model.features,
              'classifier': model.classifier,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'idx_to_class':model.class_to_idx
              }

torch.save(checkpoint, save_dir)
print("Model Saved To : "+save_dir)
