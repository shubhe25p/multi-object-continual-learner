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

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

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

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

class NaiveNet(nn.Module):
    def __init__(self):
        super(NaiveNet, self).__init__()

        # 28x28x1 => 26x26x32
        #98*98*3 => 98-5+1,98-5+1,32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        #94,94,32 => 92,92,64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.d1 = nn.Linear(92*92*64, 1024)
        self.d2 = nn.Linear(1024, 512)
        self.d3 = nn.Linear(512, 256)
        self.d4 = nn.Linear(256, 32)

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

def accuracy(true, predicted):
    corrects  = (torch.diagonal((true @ predicted.T),0) > 0.3).sum()
    accuracy = 100*corrects/8
    return accuracy


def get_accuracy(logit, target, batch_size):
        ''' Obtain accuracy for training round '''
        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects/batch_size
        return accuracy.item()
## show images

def preprocess():
    BATCH_SIZE = 32

    ## transformations
    transform = transforms.Compose(
        [transforms.ToTensor()])

    ## download and load training dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    ## download and load testing dataset
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)
    
    return testloader, trainloader

def rearrange(logits, labels):

    labelWithCosDist = torch.zeros([8,4])
    logits_detach = logits.clone().detach()
    cosDist = logits @ labels.T
    indices = torch.argmax(cosDist, dim=0)
    labelWithCosDist[indices] = logits_detach[indices]
    return labelWithCosDist

def showFrameWithPredictedBoxes(frame, coords, nameOfWindow):
    for coord in coords:
        cv2.rectangle(frame, (int(np.float32(coord[1])), int(np.float32(coord[0]))), (int(np.float32(coord[3])), int(np.float32(coord[2]))),(0,255,0))
    cv2.imshow(nameOfWindow, frame)

def saveFrameIfAccBelowThres(frame, accuracy, negSample):
    if accuracy < 20:
        negSample = torch.cat([negSample, frame])
    return negSample

def saveFrameIfAccAboveThres(frame, accuracy, posSample):
    if accuracy > 19:
        posSample = torch.cat([posSample, frame])
    return posSample
    
def getMSELoss(criterion, logits, labelWithCosDist, ewcFlag, model, negSample):
    
    loss = criterion(torch.flatten(logits), torch.flatten(labelWithCosDist))
    if ewcFlag:
        ewc = EWC(model, negSample)
        return loss + ewc.penalty(model)
    return loss

def getPR(detectedBbox, groundBbox):
    p = 0
    r = 0
    return p,r

def plotPR(p_vals, r_vals):
    plt.plot(p_vals, r_vals)
    plt.show()
    plt.xlabel("Precision")
    plt.ylabel("Recall")

def savePR(detectedBbox, groundBbox):
    p_vals=[]
    r_vals=[]
    p, r = getPR(detectedBbox, groundBbox)
    p_vals = p_vals.append(p)
    r_vals = r_vals.append(r)
    return 0,0
    

def train(model, device, fish_coords, frame, num, negSample, posSample, ewcFlag):
    learning_rate = 0.001
    num_epochs = 1

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_running_loss = 0.0
        train_acc = 0.0

        model = model.train()
        images = torch.Tensor(frame)
        images = images.to(device)
        labels = torch.Tensor(fish_coords)
        labels = labels.to(device)
        ## forward + backprop + loss
        images = torch.unsqueeze(images, dim=0)
        images = torch.permute(images, (0,3,1,2))
        logits = model(images)
        logits = torch.reshape(logits, [8,4])
        labelWithCosDist =  rearrange(logits, labels)
        loss = getMSELoss(criterion, logits, labelWithCosDist, ewcFlag, model, negSample)
        optimizer.zero_grad()
        loss.backward()
        ## update model params
        optimizer.step()
        train_running_loss += loss.detach().item()
        train_acc = accuracy(logits, labelWithCosDist)
        p_vals,r_vals = savePR(labelWithCosDist*98, labels.clone().detach()*98)
        negSample = saveFrameIfAccBelowThres(images, train_acc, negSample)
        posSample = saveFrameIfAccAboveThres(images, train_acc, posSample)
        model.eval()
        showFrameWithPredictedBoxes(frame, labelWithCosDist*98, "CL prediction")
        print('Frame: %d | Loss: %.4f | Train Accuracy: %.2f' \
              %(num, train_running_loss, train_acc))
    return negSample, posSample, p_vals, r_vals

def test(testloader, model, device, BATCH_SIZE):
    test_acc = 0.0
    for i, (images, labels) in enumerate(testloader, 0):
        images = images.to(device)
        labels = labels.to(device)  
        outputs = model(images)
        test_acc += get_accuracy(outputs, labels, BATCH_SIZE)
        
    print('Test Accuracy: %.2f'%( test_acc/i))

def detect_fish(frame, backSubKNN):
    fgMaskKNN = backSubKNN.apply(frame)
    se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
    bg=cv2.morphologyEx(fgMaskKNN, cv2.MORPH_DILATE, se)
    bg=cv2.morphologyEx(fgMaskKNN, cv2.MORPH_CLOSE, se)
    out_gray=cv2.divide(fgMaskKNN, bg, scale=255)
    out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1] 
    fish_blobs = label(out_binary)
    cv2.imshow('blob', fgMaskKNN)
    properties = ['area', 'bbox', 'bbox_area','eccentricity','centroid']
    bbox_df = pd.DataFrame(regionprops_table(fish_blobs, properties = properties))
    bbox_df = bbox_df[(bbox_df['eccentricity'] < bbox_df['eccentricity'].max()) & (bbox_df['bbox_area'] > 100) & (bbox_df['bbox_area'] < 500)]
    fish_coords = [(row['bbox-0'], row['bbox-1'],row['bbox-2'], row['bbox-3']) for index,row in bbox_df.iterrows()]
    # fish_centroid = [bbox_df['centroid'] for index,row in bbox_df.iterrows()]
    return fish_coords

def normalizeCoord(coords):
    coord_arr = np.array(coords)
    coord_arr = coord_arr/98
    return coord_arr



def main():
    
    #testloader, trainloader = preprocess()
    tracks = []
    negSample = torch.zeros([1,3,98,98])
    posSample = torch.zeros([1,3,98,98])
    cap = cv2.VideoCapture("./wells/crop-z-00.avi")
    backSubKNN = cv2.createBackgroundSubtractorKNN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NaiveNet()
    model = model.to(device)
    cnt=0
    while(True):
        ret, frame = cap.read()
        cnt=cnt+1
        cv2.imshow('Original', frame)
        cv2.waitKey(50)
        fish_coords = detect_fish(frame, backSubKNN)
        fish_coords_norm = normalizeCoord(fish_coords)
        showFrameWithPredictedBoxes(frame, fish_coords, "blob detection")
        # if fish_coords_norm.size > 0:
        #     if cnt<20000:
        #         negSample, posSample, p_vals, r_vals = train(model, device, fish_coords_norm, frame, cnt, negSample, posSample, False)
        #     else:
        #         negSample, posSample, p_vals, r_vals = train(model, device, fish_coords_norm, frame, cnt, negSample, posSample, True)
        # # test(testloader, model, device, 32, fish_coords_norm, frame)
    plotPR(p_vals, r_vals)
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()