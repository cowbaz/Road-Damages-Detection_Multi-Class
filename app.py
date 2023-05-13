### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch

from model import create_vit_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ["D00-Longitudinal Crack", "D10-Transverse Crack", "D20-Aligator Crack", "D40-Pothole",
              "D43-Cross Walk Blur", "D44-White Line Blur", "D50-Manhole Cover (TBC)"]

### 2. Model and transforms preparation ###

# Create ViT model
vit, vit_transforms = create_vit_model(
    num_classes=len(class_names), # len(class_names) would also work
)

# Load saved weights
vit.load_state_dict(
    torch.load(
        f="Pretrained_vit_feature_extractor_RDDV2AMC.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = vit_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    vit.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(vit(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings
title = "Road Damages Detection 🚧"
description = "A ViT feature extractor computer vision model to classify images of common road damages."
article = "Created at https://huggingface.co/spaces/erictam/Road-Damages-Detection_Multi-Class"

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=7, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()
