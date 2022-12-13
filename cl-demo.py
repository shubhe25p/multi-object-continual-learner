import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from skimage.measure import label, regionprops_table
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
from torch.autograd import Variable
import torch.utils.data
from matplotlib import pyplot as plt
from podm.metrics import get_pascal_voc_metrics
from sklearn.metrics import precision_score, recall_score

# This is a function that creates a `Variable` object from a `torch.Tensor` object.
# The `Variable` class is a wrapper class for tensors that allows the tensor to be differentiated
# and includes additional functionality for the deep learning framework PyTorch.

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    # The `use_cuda` parameter determines whether the `Variable` should be created on the GPU if one is available.
    # If `True` and a GPU is available, the `Variable` will be created on the GPU.
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()

    # The `**kwargs` parameter allows for the passing of additional keyword arguments to the `Variable` constructor.
    return Variable(t, **kwargs)

class EWC(object):
     # This initializes the EWC object. The `model` parameter is a PyTorch `nn.Module` containing the model
    # to be regularized. The `dataset` parameter is a list of input/output pairs used to compute the diagonal
    # Fisher information matrix.
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset
        # The `params` attribute contains the parameters of the `model` that require gradient computations
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    # computes the fisher information matrix
    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = variable(input)
            output = self.model(torch.unsqueeze(input, dim=0))
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    # calculates the penalty/regularizer which will be added to the loss
    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

class NaiveNet(nn.Module):
    def __init__(self):
        super(NaiveNet, self).__init__()
        # a simple neural network with 2conv layers and 4 FC layers
        # 28x28x1 => 26x26x32
        #98*98*3 => 98-5+1,98-5+1,32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        #94,94,32 => 92,92,64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.d1 = nn.Linear(92*92*64, 1024)
        self.d2 = nn.Linear(1024, 512)
        self.d3 = nn.Linear(512, 256)
        self.d4 = nn.Linear(256, 32)

    # forward pass of the model, it uses relu as the activation function and outputs a sigmoid which is considered as a normalized output.
    def forward(self, x):
        # 32x1x28x28 => 32x32x26x26
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # flatten => 32 x (32*26*26)
        x = x.flatten(start_dim = 1)

        # 32 x (32*26*26) => 32x128
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        x = F.relu(x)
        x = self.d3(x)
        x = F.relu(x)
        # logits => 32x10
        logits = self.d4(x)
        out = torch.sigmoid(logits)
        return out

## functions to show an image
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# predict the accuracy of the model
def accuracy(true, predicted):
    corrects  = (torch.diagonal((true @ predicted.T),0) > 0.3).sum()
    accuracy = 100*corrects/8
    return accuracy

def rearrange(logits, labels):
    # Create a tensor with 8 rows and 4 columns to hold the logits
    # for each label, with the cosine distance for each
    labelWithCosDist = torch.zeros([8,4])

    # Make a copy of the logits tensor and detach it from the computation graph
    logits_detach = logits.clone().detach()

    # Compute the cosine distance between the logits and labels
    cosDist = logits @ labels.T

    # Get the indices of the maximum values in each column of the cosine distance
    indices = torch.argmax(cosDist, dim=0)

    # Set the values of the labelWithCosDist tensor at the indices to the corresponding
    # logits values
    labelWithCosDist[indices] = logits_detach[indices]

    # Return the final labelWithCosDist tensor
    return labelWithCosDist


def showFrameWithPredictedBoxes(frame, coords, centroids, nameOfWindow):
    # Loop over each pair of coordinates and centroids
    for i, (coord, cent) in enumerate(zip(coords, centroids, strict=True)):
        # Draw a rectangle on the frame using the coordinates
        cv2.rectangle(frame, (int(np.float32(coord[1])), int(np.float32(coord[0]))), (int(np.float32(coord[3])), int(np.float32(coord[2]))),(0,255,0))

        # Draw a circle on the frame at the centroid coordinates
        cv2.circle(frame, (int(np.float32(cent[1])),int(np.float32(cent[0]))), radius=0, color=(0, 0, 255), thickness=-10)

        # Write the index of the object on the frame at the top left corner of the bounding box
        cv2.putText(frame,str(i+1), (int(coord[1]),int(coord[0])),0, 0.5, (0,0,0),2)

    # Show the frame with the predicted bounding boxes and object indices in a window with the given name
    cv2.imshow(nameOfWindow, frame)



def showFrameWithPredictedBoxesCL(frame, coords, nameOfWindow):
    # same functionality as the above code
    for i, coord in enumerate(coords):
        cv2.rectangle(frame, (int(np.float32(coord[1])), int(np.float32(coord[0]))), (int(np.float32(coord[3])), int(np.float32(coord[2]))),(0,255,0))
        cv2.putText(frame,str(i+1), (int(coord[1]),int(coord[0])),0, 0.5, (0,0,0),2)
    cv2.imshow(nameOfWindow, frame)


def saveFrameIfAccBelowThres(frame, accuracy, negSample):
    # used to create the negative sample sets which are further used for retraining
    if accuracy < 20:
        negSample = torch.cat([negSample, frame])
    return negSample

def saveFrameIfAccAboveThres(frame, accuracy, posSample):
    # used to create the positive sample sets which are further used for retraining
    if accuracy > 19:
        posSample = torch.cat([posSample, frame])
    return posSample
    
def getMSELoss(criterion, logits, labelWithCosDist, ewcFlag, model, negSample):
    
    # calculates the MSE loss with EWC update
    # If the EWC flag is true than the corresponding penalty is added to the loss
    # if not normal loss is returned
    loss = criterion(torch.flatten(logits), torch.flatten(labelWithCosDist))
    if ewcFlag:
        ewc = EWC(model, negSample)
        return loss + ewc.penalty(model)
    return loss



def plotPR(p_vals, r_vals):
    # plotting PR values
    plt.plot(p_vals, r_vals)
    plt.show()
    plt.xlabel("Precision")
    plt.ylabel("Recall")

def savePR(detectedBbox, groundBbox):
    # utilize sklearn metrics to find the precision and reacall of the predicted and groundtruth bboxes
    p = precision_score(groundBbox,detectedBbox)
    r = recall_score(groundBbox, detectedBbox)
    p_vals = p_vals.append(p)
    r_vals = r_vals.append(r)
    return p_vals, r_vals
    

def train(model, device, fish_coords, frame, num, negSample, posSample, ewcFlag):
    
    # set learning rate and number of epochs
    learning_rate = 0.001
    num_epochs = 1

    # initialize mean square loss and use adam optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(num_epochs):
        train_running_loss = 0.0
        train_acc = 0.0

        model = model.train()
        # convert frame to tensor object
        images = torch.Tensor(frame)
        images = images.to(device)
        # convert fish_coords to tensor object
        labels = torch.Tensor(fish_coords)
        labels = labels.to(device)
        # unsqueeze will add an extra dimension
        images = torch.unsqueeze(images, dim=0)
        # permute will rearrange the images tensor
        images = torch.permute(images, (0,3,1,2))
        # permuted tensor frame makes a forward pass
        logits = model(images)
        # output logits are bounding box coordinates
        logits = torch.reshape(logits, [8,4])
        # rearrange logits based on the cosine distance
        labelWithCosDist =  rearrange(logits, labels)
        loss = getMSELoss(criterion, logits, labelWithCosDist, ewcFlag, model, negSample)
        optimizer.zero_grad()
        loss.backward()
        ## update model params
        optimizer.step()
        train_running_loss += loss.detach().item()
        # get accuracy of the model while its training
        train_acc = accuracy(logits, labelWithCosDist)
        # seggregate sample set based on the accuracy threshold
        negSample = saveFrameIfAccBelowThres(images, train_acc, negSample)
        posSample = saveFrameIfAccAboveThres(images, train_acc, posSample)
        model.eval()
        # uncomment below code to show predictions by CL, please wait it will take some time
        showFrameWithPredictedBoxesCL(frame, labelWithCosDist*98, "CL prediction")
        print('Frame: %d | Loss: %.4f | Train Accuracy: %.2f' \
              %(num, train_running_loss, train_acc))
    return negSample, posSample

# Defines a function that takes as input a data loader for a test set, a trained model, the device used for training and inference, and the batch size for the test set
def test(testloader, model, device, BATCH_SIZE):
    # Initialize a variable to store the accuracy of the model on the test set
    test_acc = 0.0
    
    # For each batch of data in the test set
    for i, (images, labels) in enumerate(testloader, 0):
        # Move the data to the device specified
        images = images.to(device)
        labels = labels.to(device)
        
        # Get the model's predictions on the batch of data
        outputs = model(images)
        
        # Update the test accuracy by computing the accuracy of the model's predictions on the current batch of data
        test_acc += accuracy(outputs, labels)
        
    # Compute the average test accuracy and print it
    print('Test Accuracy: %.2f'%( test_acc/i))

# Defines a function that takes as input a video frame and a pre-trained KNN background subtraction model
def detect_fish(frame, backSubKNN):
    # Apply the KNN background subtraction model to the frame to get a mask highlighting the fish
    fgMaskKNN = backSubKNN.apply(frame)
    
    # Use morphological transformations to clean up the mask
    se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
    bg=cv2.morphologyEx(fgMaskKNN, cv2.MORPH_DILATE, se)
    bg=cv2.morphologyEx(fgMaskKNN, cv2.MORPH_CLOSE, se)
    
    # Identify individual fish in the mask using connected component analysis
    fish_blobs = label(fgMaskKNN)
    
    # Show the mask with the identified fish in a window
    cv2.imshow('blob', fgMaskKNN)
    
    # Compute the properties of each fish
    properties = ['area', 'bbox', 'bbox_area','eccentricity','centroid']
    bbox_df = pd.DataFrame(regionprops_table(fish_blobs, properties = properties))
    
    # Select only the fish that meet certain criteria (eccentricity, bounding box area, etc.)
    bbox_df = bbox_df[(bbox_df['eccentricity'] < bbox_df['eccentricity'].max()) & (bbox_df['bbox_area'] > 100) & (bbox_df['bbox_area'] < 500)]
    
    # Extract the coordinates of the centroid and bounding box of each fish
    fish_centroid = [(row['centroid-0'], row['centroid-1']) for index,row in bbox_df.iterrows()]
    fish_coords = [(row['bbox-0'], row['bbox-1'],row['bbox-2'], row['bbox-3']) for index,row in bbox_df.iterrows()]
    
    
    return fish_coords, fish_centroid

def normalizeCoord(coords):
    # normalize the coordinates by the width and height of each well(98x98)
    coord_arr = np.array(coords)
    coord_arr = coord_arr/98
    return coord_arr

# This function calculates the cost of tracks between the current and previous frames
# The cost is calculated as the Euclidean distance between the current and previous centroids
def costOfTracks(centroids, cost, tracks):
    # If there is more than one track
    if len(tracks) > 1:
        # Initialize a matrix to store the costs for each centroid-track pair
        costCurrPrev = np.zeros((len(centroids), len(tracks[-1])));
        # Loop through each centroid and track
        for i, curr in enumerate(centroids):
            for j, prev in enumerate(tracks[-1]):
                # Calculate the cost as the Euclidean distance between the centroid and track
                costCurrPrev[i][j] = ((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)**0.5
        # Append the calculated costs to the cost array
        cost.append(costCurrPrev)
    # Return the updated cost array
    return cost


def main():
    
    #stores centroids of all previous frames
    tracks = []
    costTrackToDetection = []
    # initialize sample sets 
    negSample = torch.zeros([1,3,98,98])
    posSample = torch.zeros([1,3,98,98])
    cap = cv2.VideoCapture("./wells/crop-z-00.avi")
    backSubKNN = cv2.createBackgroundSubtractorKNN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # initialize conv model
    model = NaiveNet()
    model = model.to(device)
    # cnt represents the number of frames
    cnt=0
   # Continuously capture frames from the video
    while(True):
        # Capture the current frame
        ret, frame = cap.read()
        # Increment the frame counter
        cnt=cnt+1
        # Show the current frame
        cv2.imshow('Original', frame)
        # Wait for 50 milliseconds
        cv2.waitKey(50)
        # Detect fish in the current frame
        fish_coords, fish_centroid = detect_fish(frame, backSubKNN)
        # If fish are detected, add their centroids to the tracks array
        if fish_centroid:
            tracks.append(fish_centroid)
        # Calculate the cost of each track in the previous frame to the centroids in the current frame
        costTrackToDetection = costOfTracks(fish_centroid, costTrackToDetection, tracks)
        # If there is more than one cost, perform the Hungarian algorithm to calculate the assignment
        # of tracks to centroids
        if len(costTrackToDetection) > 1:
            _,assignment = linear_sum_assignment(costTrackToDetection[-1])
        # Normalize the coordinates of the detected fish
        fish_coords_norm = normalizeCoord(fish_coords)
        # Show the current frame with bounding boxes around the detected fish
        showFrameWithPredictedBoxes(frame, fish_coords, fish_centroid, "blob detection")
        # If fish are detected, train the model on the normalized coordinates
        if fish_coords_norm.size > 0:
            # If more than 2000 frames have been captured, train the model on the normalized coordinates
            if cnt>200 and cnt<2000:
                negSample, posSample = train(model, device, fish_coords_norm, frame, cnt, negSample, posSample, False)
            # If more than 20000 frames have been captured, train the model on the normalized coordinates
            # with ewc penalty
            if cnt>20000:
                negSample, posSample = train(model, device, fish_coords_norm, frame, cnt, negSample, posSample, True)

    p_vals, r_vals = savePR()
    plotPR(p_vals, r_vals)
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()