#%%
import json

import pandas as pd
df = pd.read_json("Batch2__BioSAV_BIofiltration_18mois_05frame3049.jpg.json", orient='index')

object_list = df.values[3][0]

boxes_list = []

for item in object_list:
    if item['classTitle'] == 'People':
        boxes_list.append(item['points']['exterior'])
        
def get_boxes(json_file):
    df = pd.read_json(json_file, orient='index')

    object_list = df.values[3][0]

    box_list = []

    for item in object_list:
        if item['classTitle'] == 'People':

            box_list.append(item['points']['exterior'][0]+item['points']['exterior'][1])
    return box_list

# %%
from PIL import Image
from torchvision import models
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import matplotlib.pyplot as plt
import cv2




# %%
# Loading the model and the dataset
# Loads pretrained VGG model and sets it to eval mode

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = model.eval()
# %%
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# %%
def get_prediction(img_path, threshold):
    img = Image.open(img_path)  # Load the image
    transform = T.Compose([T.ToTensor()])  # Defing PyTorch Transform
    img = transform(img)  # Apply the transform to the image
    pred = model([img])  # Pass the image to the model
    
    pred_class = [
        COCO_INSTANCE_CATEGORY_NAMES[i]
        for i in list(pred[0]["labels"].numpy())
    ]  # Get the Prediction Score
    pred_boxes = [
        [(int( i[0] ) , int( i[1] ) ), (int( i[2] ), int( i[3] ) )]
        for i in list(pred[0]["boxes"].detach().numpy())
    ]  # Bounding boxes
    pred_score = list(pred[0]["scores"].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][
        -1
    ]  # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[: pred_t + 1]
    pred_class = pred_class[: pred_t + 1]
    return pred_boxes, pred_class


# %%
def object_detection_api(
    img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3
):
    boxes, pred_cls = get_prediction(img_path, threshold)  # Get predictions
    img = cv2.imread(img_path)  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    for i in range(len(boxes)):
        if pred_cls[i] == "person":
            cv2.rectangle(
                img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th
                )   # Draw Rectangle with the coordinates
            cv2.putText(
                img,
                pred_cls[i],
                boxes[i][0],
                cv2.FONT_HERSHEY_SIMPLEX,
                text_size,
                (0, 255, 0),
                thickness=text_th,
                )  # Write the prediction class
    plt.figure(figsize=(20, 30))  # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# %%
# Try the detection model for the image of your choice
# Example to help, if I have a folder named data with a jpeg format picture called test, the result would be:
object_detection_api("Batch2__BioSAV_BIofiltration_18mois_05frame3049.jpg")


# %%

def afficher_boxes_json(
    img_path, boxes, rect_th=3, text_size=3, text_th=3
):
    img = cv2.imread(img_path)  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    for i in range(len(boxes)):
        cv2.rectangle(
            img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th
        )  # Draw Rectangle with the coordinates
    plt.figure(figsize=(20, 30))  # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# %% https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

#%% Definition of the dataset ,marche pas pour l'instant
import torch
import os
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.jsons = list(sorted(os.listdir(os.path.join(root, "jsons"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        json_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        boxes=get_boxes(json_path)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        
        # get bounding box coordinates for each mask
        num_objs = len(boxes)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

#def critere


def prop_people_detected(
        img_path, threshold=0.5
        ):
    
    boxes_person = []
    
    # On recupere la liste des personnes identifiees par l'algo
    boxes, pred_cls = get_prediction(img_path, threshold)
    
    for i in range(len(boxes)):
        if pred_cls[i] == "person":
            boxes_person.append(boxes[i])
    
    
    # On recupere la liste des personnes reelles
    df = pd.read_json(img_path+".json", orient='index')
    object_list = df.values[3][0]
    real_boxes_list = []
    
    for item in object_list:
        if item['classTitle'] == 'People':
            real_boxes_list.append(item['points']['exterior'])
            # real_boxes_list contient donc tous les rectangles voulus
    

