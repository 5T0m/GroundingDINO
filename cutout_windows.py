import os
import sys
import supervision as sv
import torch
from groundingdino.util.inference import Model
import numpy as np
import cv2
from typing import List


### Helper Functions
def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

def is_within(box1, box2):
    """
    Check if box1 lies completely within box2.

    Args:
        box1 (list or tuple): [x1, y1, x2, y2] coordinates of the first bounding box.
        box2 (list or tuple): [x1, y1, x2, y2] coordinates of the second bounding box.

    Returns:
        bool: True if box1 lies completely within box2, False otherwise.
    """
    return (
        box1[0] >= box2[0] and  # box1's left edge is to the right of or equal to box2's left edge
        box1[1] >= box2[1] and  # box1's top edge is below or equal to box2's top edge
        box1[2] <= box2[2] and  # box1's right edge is to the left of or equal to box2's right edge
        box1[3] <= box2[3]      # box1's bottom edge is above or equal to box2's bottom edge
    )

#increase the margins of the bounding box
def adjust_bounding_box(bounding_box,percentage, image_width, image_height):
    x1, y1, x2, y2 = bounding_box
    width = x2 - x1
    height = y2 - y1

    # Increase margins by percentage%
    x1_new = max(0, x1 - percentage * width)  # Ensure x1 is not less than 0
    y1_new = max(0, y1 - percentage * height) # Ensure y1 is not less than 0
    x2_new = min(image_width, x2 + percentage * width)  # Ensure x2 does not exceed image width
    y2_new = min(image_height, y2 + percentage * height) # Ensure y2 does not exceed image height

    adjusted_box = [int(x1_new), int(y1_new), int(x2_new), int(y2_new)]
    return adjusted_box

### End Helper functions

GROUNDING_DINO_CONFIG_PATH = './groundingdino/config/GroundingDINO_SwinT_OGC.py'
GROUNDING_DINO_CHECKPOINT_PATH = './weights/groundingdino_swint_ogc.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

IMAGES_DIRECTORY = './windows'
IMAGES_EXTENSIONS = ['jpg', 'jpeg', 'png']

CLASSES = ['window']
BOX_TRESHOLD = 0.30
TEXT_TRESHOLD = 0.07


# Extract labels from images
images = {}
annotations = {}

image_paths = sv.list_files_with_extensions(
    directory=IMAGES_DIRECTORY,
    extensions=IMAGES_EXTENSIONS)

for image_path in image_paths:
    image_name = image_path.name
    image_path = str(image_path)
    image = cv2.imread(image_path)

    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    detections = detections[detections.class_id != None]
    images[image_name] = image
    annotations[image_name] = detections


# Creating cutouts

# Set margin to more than 0 to get larger cutout than bounding boxes.
margin = 0.05

for image_name, detections in annotations.items():
    image = images[image_name]
    im_width = image.shape[1]
    im_height = image.shape[0]

    bounding_boxes = detections.xyxy

    # Delete bounding boxes contained in larger boxes
    indices_to_delete = []
    for i, box1 in enumerate(bounding_boxes):
        for j, box2 in enumerate(bounding_boxes):
            if i != j and is_within(box1,box2): 
                indices_to_delete.append(i)
    
    new_bounding_boxes = []
    for i, box in enumerate(bounding_boxes):
        if i not in indices_to_delete:
            new_bounding_boxes.append(box)
            
    # Loop through each bounding box and create cutouts
    for i, box in enumerate(new_bounding_boxes):
        new_image = image

        # Increase the size cutout margin of he bounding box
        box = adjust_bounding_box(box,0.05, im_width,im_height)
    
        x1, y1, x2, y2 = box
 
        # Crop the region of interest
        cropped_image = new_image[y1:y2, x1:x2]

        # Save the cropped image
        if cropped_image.size > 0:
            cropped_width = cropped_image.shape[0]
            cropped_height = cropped_image.shape[1]
            aspect_ratio = cropped_width / cropped_height

            min_aspect_ratio = 0.1
            max_aspect_ratio = 10
            
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                output_path = f"./cutouts/cutout_{image_name[:-4]}_{i}.jpg"
                cv2.imwrite(output_path, cropped_image)

print("Cutouts saved successfully!")





