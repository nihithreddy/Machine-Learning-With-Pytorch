from helper import *
#Creating an object of ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument('image_path',help = 'Path of the image to be predicted')
parser.add_argument('checkpoint',help = 'Path of the Checkpoint file')
parser.add_argument('--top_k',help = 'Number specifying how many top predictions to find')
parser.add_argument('--gpu',help = 'Use GPU for training')
parser.add_argument('--category_names',help= 'Path of the json file for mapping')
parser.set_defaults(gpu=False)

#Parsing the Arguments
args = parser.parse_args()
image_path = args.image_path
check_point = args.checkpoint
top_k = args.top_k
device = args.gpu
category_names = args.category_names

#Checking the values provided by the user
if(top_k is None):
    top_k = 5
else:
    top_k = int(top_k)

if(device == False):
    device = "cpu"
else:
    if(torch.cuda.is_available()):
        device = "cuda"
    else:
        Print("Torch Cuda is not available!! Hence Using CPU")
        device = "cpu"

if(category_names is None):
    category_names = 'cat_to_name.json'
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    file_name,file_extension = os.path.splitext(category_names)
    if(file_extension != 'json'):
        print("Please use a file with a json extension")
        exit()
    else:
        with open(category_names,'r') as f:
            cat_to_name = json.load(f)
if((image_path == None) or (check_point == None)):
    print("Image Path and Check Point cannot be None")
    exit()
    
#print(top_k)
#print(device)
#print(check_point)
#print(image_path)
#print(category_names)
                    
#Defining the no of units in the output layer
output_units = 102

#A function that loads a checkpoint and rebuilds the model
def load_model(path):
    model_state = torch.load(path)
    arch = model_state['transfer_model']
    hidden_units = 256
    if(arch == "vgg13"):
        model = models.vgg13(pretrained = True)
        input_units = 25088
        hidden_units = 4096
    else:
        model = models.densenet121(pretrained = True)
        input_units = 1024
        hidden_units = 500
    model.classifier = nn.Sequential(nn.Linear(input_units,hidden_units),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2),
                                    nn.Linear(hidden_units,output_units),
                                    nn.LogSoftmax(dim = 1))
    model.load_state_dict(model_state['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    edit_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.RandomCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    pil_image = edit_transform(pil_image)
    final_image = np.array(pil_image)
    final_image = final_image.transpose((0,2,1))
    return final_image
    


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.class_to_idx = image_datasets['train_data'].class_to_idx
    model.to(device)
    img = process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = img.unsqueeze(0)
    img  = img.float()
    if(device =="cpu"):
        img = img.cpu()
    else:
        img = img.cuda()
        
    with torch.no_grad():
        prediction = model.forward(img)
        
    ps = torch.exp(prediction)
    top_p,top_class = ps.topk(topk,dim = 1)
    return top_p[0].cpu().numpy(),top_class[0].cpu().numpy()


# TODO: Display an image along with the top 5 classes
def display(image_path, model):
    
    # Setting plot area
    plt.figure(figsize = (3,6))
    ax = plt.subplot(2,1,1)
    
    # Display test flower
    img = process_image(image_path)
    image_parts  = image_path.split('/')
    #print(image_parts)
    #print(cat_to_name[image_parts[2]])
    imshow(img, ax, title = cat_to_name[image_parts[2]])
    
    # Making prediction
    probs, classes = predict(image_path, model) 
    labels = [cat_to_name[str(label)] for label in classes]
    #print(labels)
    fig,ax = plt.subplots(figsize=(4,3))
    sticks = np.arange(len(classes))
    ax.barh(sticks, probs, height=0.3, linewidth=2.0, align = 'center')
    ax.set_yticks(ticks = sticks)
    ax.set_yticklabels(labels)
    

model = load_model(check_point)
image_title  = image_path.split('/')
print("Test image:" + cat_to_name[image_title[2]])
display(image_path, model)
print("Prediction result:")
scores, classes = predict(image_path, model)
flowers = [cat_to_name[str(i)] for i in classes]
print(scores)
print(flowers)
