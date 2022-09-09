import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from PIL import Image
from torchvision import models
import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import matplotlib.pyplot as plt
import cv2

import pandas as pd


# Return a list of boxes  in [x0,y0,x1,y1] format
def get_boxes(json_file):
    df = pd.read_json(json_file, orient='index')

    object_list = df.values[3][0]

    box_list = []

    for item in object_list:
        if item['classTitle'] == 'People':
            box_list.append(item['points']['exterior'][0] + item['points']['exterior'][1])
    return box_list

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

#Create a model from fasterrcnn_resnet50_fpn with a chosen number of outputs
def get_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    # models.detection.maskrcnn_resnet50_fpn(weights= )
    model = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model



# %% Definition of the dataset

class WorksiteDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGimages"))))
        self.jsons = list(sorted(os.listdir(os.path.join(root, "JSONfiles"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGimages", self.imgs[idx])
        json_path = os.path.join(self.root, "JSONfiles", self.jsons[idx])

        img = Image.open(img_path).convert("RGB")
        boxes = get_boxes(json_path)

        num_box = len(boxes)
        if num_box > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            # negative example, ref: https://github.com/pytorch/vision/issues/2144
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_box,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_box,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img,target)
        return img, target

    def __len__(self):
        return len(self.imgs)


# %%

from engine import train_one_epoch, evaluate
import utils

#returns a tune refined model from fasterrcnn_resnet50_fpn
def main():
    print("main")
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = WorksiteDataset('Detection_Train_Set', get_transform(train=True))
    dataset_test = WorksiteDataset('Detection_Train_Set', get_transform(train=False))
    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
    return model

#To execute to train the model
#if __name__ == "__main__":
#    model=main()
#    torch.save(model.state_dict(), "modell10.pth")

#%%
#modell10.pth has been trained 10 epochs on the dataset
modell=get_model(2)
modell.load_state_dict(torch.load("modell10.pth",map_location=torch.device('cpu')))
modell.eval()

def get_paths(dataset,idx):
    img_path= dataset.root + "/PNGimages/"+dataset.imgs[idx]
    json_path = dataset.root + "/JSONfiles/" + dataset.jsons[idx]
    return img_path,json_path
def get_prediction(model,datas,idx, threshold):

    img,target = datas[idx]  # Apply the transform to the image
    pred = model([img])  # Pass the image to the model
    NAMES=["background","person"]

    pred_class = [
        NAMES[i]
        for i in list(pred[0]["labels"].numpy())
    ]  # Get the Prediction Score
    pred_boxes = [
        [(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]
        for i in list(pred[0]["boxes"].detach().numpy())
    ]  # Bounding boxes
    pred_score = list(pred[0]["scores"].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][
        -1
    ]  # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[: pred_t + 1]
    pred_class = pred_class[: pred_t + 1]
    return pred_boxes, pred_class

#Takes a model, a dataset and an id, shows the picture corresponding to the id in the dataset
# with the boxes guessed by the model in red, and the boxes from the json file in green
def object_detection_api(
        model,dataset,idx, threshold=0.5, rect_th=3, text_size=3, text_th=3
):
    img_path,json_path=get_paths(dataset,idx)
    boxes, pred_cls = get_prediction(model,dataset,idx, threshold)  # Get predictions
    img = cv2.imread(img_path)  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    boxesjson=get_boxes(json_path)
    for i in range(len(boxes)):
        if pred_cls[i] == "person":
            cv2.rectangle(
                img, boxes[i][0], boxes[i][1], color=(255, 0, 0), thickness=rect_th
            )  # Draw Rectangle with the coordinates
    for i in range(len(boxesjson)):
        cv2.rectangle(
            img, (boxesjson[i][0],boxesjson[i][1]),(boxesjson[i][2],boxesjson[i][3]), color=(0, 255, 0), thickness=2
        )  # Draw Rectangle with the coordinates

    plt.figure(figsize=(20, 30))  # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

WSdataTrain=WorksiteDataset("Detection_Train_Set",get_transform(train=False))
WSdataTest=WorksiteDataset("Detection_Test_Set",get_transform(train=False))

#object_detection_api(modell,WSdataTest,3)
