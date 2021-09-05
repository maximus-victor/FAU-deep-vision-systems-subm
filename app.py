"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image
import streamlit as st


import pandas as pd
import numpy as np
import glob
import joblib
import os
from pathlib import Path


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import pickle
from skimage.color import rgb2gray
import skimage

from imutils import paths
from sklearn import preprocessing
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

import matplotlib.patches as patches
from skimage.color import gray2rgb
import engine
import utils
import cocoeval
import coco_eval
import coco_utils
import transforms

from skimage.color import rgb2gray, gray2rgb

import torch
from torchvision import transforms, datasets, models
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
#import utils
import pickle
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from html_mardown import app_off,app_off2, model_predicting, loading_bar, result_pred, image_uploaded_success, more_options, class0, class1, class2, class3, class4, s_load_bar, class0_side, class1_side, class2_side, class3_side, class4_side, unknown, unknown_side, unknown_w, unknown_msg
from datasets_streamlit import WaferDataset, classes
from models_streamlit import RegNet, get_faster_rcnn

DIR_DATEN = "/content/drive/MyDrive/DeepVis/data"   # make sure to always adapt this to your own folser structure
DIR_DATEN_01 = os.path.join(DIR_DATEN, "01_Daten")
DIR_WAFER_IMAGES = os.path.join(DIR_DATEN, "WaferImages")
lb = pickle.load(open('/content/drive/MyDrive/DEEPVIS/label_encoder.pkl', 'rb'))
lb_OD = pickle.load(open('/content/drive/MyDrive/DEEPVIS/label_encoder_OD.pkl', 'rb'))

# Load OD
model_OD_path = "/content/drive/MyDrive/DEEPVIS/models/od_faster_rcnn6classes.pth"
model_OD = get_faster_rcnn()

model_OD.load_state_dict(torch.load(Path(model_OD_path), map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')), strict=True)
states_OD = torch.load(Path(model_OD_path), map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

#DataLoader for pytorch dataset
def Loader(img_path=None,uploaded_image=None, uploaded_state=False, demo_state=True, model='RegNet'):
    test_dataset = WaferDataset(path=img_path, model=model, uploaded_image=uploaded_image, uploaded_state=uploaded_state, demo_state=demo_state)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    return test_loader
    
#@st.cache(allow_output_mutation=True, max_entries=10, ttl=3600)
def inference(model, states, img, device):
    model.to(device)
    probs = []
    img = img.to(device)
    avg_preds = []
    with torch.no_grad():
        y_pred = model(img)
    _,pred = torch.max(y_pred, dim=1)
    return pred


#Set App title
st.title('Solar Cell Web App')
#App description
st.write("The app classify the error state of a solar cell.")
st.markdown('***')

#model= "saved_models/" + style_name + ".pth"
model_name = "RegNet"
model = "/content/drive/MyDrive/DEEPVIS/models/" + str(model_name)
model_OD_path = "/content/drive/MyDrive/DEEPVIS/models/faster_rcnn_model.pth"

uploaded_image = None
decision_boundary = 0.8


# get all the image paths
image_paths = list(paths.list_images(DIR_WAFER_IMAGES))

#Hide warnings
st.set_option("deprecation.showfileUploaderEncoding", False)

#Set the directory path
my_path= '/content/drive/MyDrive/DEEPVIS/'

with open('/content/drive/MyDrive/DEEPVIS/dataset_dict.pickle', 'rb') as file:
    dataset_dict = pickle.load(file)
X_test = dataset_dict['X_test']
y_test = dataset_dict['y_test']


random3 = np.random.randint(0,len(X_test))

img_1_path= X_test.iloc[1]
img_2_path= X_test.iloc[2]
img_3_path= X_test.iloc[3]
banner_path= my_path + '/images/banner.png'


#Read and display the banner
st.sidebar.image(banner_path,use_column_width=True)


#Set the selectbox for demo images
st.write('**Select a model for Image Classification**')
menu_model = ['Choose ..','RegNet', 'VGG16', 'ResNet']
choice = st.selectbox('Select a model', menu_model)


CHOICES_IMG = {0: 'Choose ..', img_1_path: 'Image 1', img_2_path: 'Image 2', img_3_path: 'Image 3'}

def format_func_img(sel_img_path):
    return CHOICES_IMG[sel_img_path]

sel_img_path = st.selectbox("Select a test image", options=list(CHOICES_IMG.keys()), format_func=format_func_img)

#Set the box for the user to upload an image
st.write("**Upload your an image**")
uploaded_image = st.file_uploader("Upload your image in JPG or PNG format", type=["jpg", "png", "tiff", "tif"])

decision_boundary = st.slider("Decision threshold of Object Detection", min_value=0., max_value=1.0, step=0.1, value=0.8)

#Function to deploy the model and print the report
def deploy(file_path=None,uploaded_image=uploaded_image, uploaded_state=False, demo_state=True) :
    #Load the model and the weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegNet()
    model.load_state_dict(torch.load(Path(my_path + 'models/RegNet.pth'), map_location=device), strict=True)

    states = torch.load(Path(my_path + 'models/RegNet.pth'), map_location=device)
    #model.load_state_dict(states)
        
    #Display the uploaded/selected image
    st.markdown(model_predicting, unsafe_allow_html=True)
    if demo_state:
        test_loader= Loader(img_path=file_path, uploaded_image=None, uploaded_state=False, demo_state=True) 
        image_1 = cv2.imread(file_path)
    if uploaded_state:
        test_loader= Loader(img_path=None, uploaded_image=uploaded_image, uploaded_state=True, demo_state=False)
        image_1 = plt.imread(uploaded_image)
    
    #for img in test_loader:
    #Inference
    pred = inference(model, states, next(iter(test_loader)), device)
    #pred_idx = pred.to('cpu').numpy().argmax(1)

    st.write("Prediction - Index:", pred[0], "-> Label:", lb.inverse_transform(list(pred))[0])
    
def apply_gradCAM(img_path=None,uploaded_image=None, uploaded_state=False, demo_state=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegNet()
    model.load_state_dict(torch.load(Path(my_path + 'models/RegNet.pth'), map_location=device), strict=True)
    target_layer = model.conv3
    if demo_state:
        test_loader= Loader(img_path=img_path, uploaded_image=None, uploaded_state=False, demo_state=True) 
    if uploaded_state:
        test_loader= Loader(img_path=None, uploaded_image=uploaded_image, uploaded_state=True, demo_state=False)

    input_tensor = next(iter(test_loader))
    
    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!
    
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=torch.cuda.is_available())
    
    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    target_category = None
    
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    if demo_state:
        visualization = show_cam_on_image((cv2.resize(cv2.imread(img_path), (256,256))/255.).astype(np.float32), grayscale_cam, use_rgb=False)
    if uploaded_state:
        img = gray2rgb(np.array(plt.imread(uploaded_image)))

        visualization = show_cam_on_image((cv2.resize(img, (256,256))/255.).astype(np.float32), grayscale_cam, use_rgb=False)

    st.image(visualization, width=256, channels='BGR')
    
def bounding_box_prediction(pretrained_model, img_path, detection_threshold=0.8, uploaded_image=None, uploaded_state=False, demo_state=True):

    if demo_state:
        test_loader= Loader(img_path=img_path, uploaded_image=None, uploaded_state=False, demo_state=True,  model="ObjectDetect")
    if uploaded_state:
        test_loader= Loader(img_path=None, uploaded_image=uploaded_image, uploaded_state=True, demo_state=False,  model="ObjectDetect")

    input_tensor = next(iter(test_loader)) 
    
    runtime = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = pretrained_model
    model = model.to(runtime)

    img = input_tensor
    img = img.to(runtime)

    thr = detection_threshold
    
    model.eval()

    with torch.no_grad():
    
        img_pred = model(img)

        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in img_pred]

        if len(outputs[0]['boxes']) != 0:
            for counter in range(len(outputs[0]['boxes'])):
                boxes = outputs[0]['boxes'].data.numpy()
                labels = lb_OD.inverse_transform(list((outputs[0]['labels'])-1))
                scores = outputs[0]['scores'].data.numpy()
                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                labels = labels[scores >= detection_threshold]
    
                draw_boxes = zip(boxes.copy(), labels.copy())
                #target_boxes = targets[0]['boxes']
          
            orig_image = img.cpu()
            orig_image = np.asarray(orig_image).squeeze().transpose(1,2,0)
            orig_image = gray2rgb(orig_image).squeeze()

        fig, ax = plt.subplots()
        ax.imshow(orig_image)
        for box, pred_label in draw_boxes:

            rect = patches.Rectangle((box[0], box[1]), (box[2]-box[0]), (box[3]-box[1]), linewidth=3, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            centerx = box[2] -(box[2]-box[0])  # obviously use a different formula for different shapes
            centery = box[3] +12 # obviously use a different formula for different shapes
    
            t = plt.text(centerx, centery, str(pred_label))
            t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white', pad= 0.5))
        #for box in target_boxes:
        #  rect = patches.Rectangle((box[0], box[1]), (box[2]-box[0]), (box[3]-box[1]), linewidth=1, edgecolor='g', facecolor='none')
        #  ax.add_patch(rect)
        plt.axis("off")
        #plt.show()
        st.pyplot(fig)



def show_img(file_path=None,uploaded_image=uploaded_image, uploaded_state=False, demo_state=True):
    if demo_state:
        if isinstance(file_path, str):
            test_loader = Loader(img_path=file_path, uploaded_image=None, uploaded_state=False, demo_state=True) 
            image_1 = cv2.imread(file_path)
            st.sidebar.markdown(image_uploaded_success, unsafe_allow_html=True)
            st.sidebar.image(image_1, width=301, channels='BGR')
    if uploaded_state:
        test_loader = Loader(img_path=None, uploaded_image=uploaded_image, uploaded_state=True, demo_state=False)
        image_1 = plt.imread(uploaded_image)
        st.sidebar.markdown(image_uploaded_success, unsafe_allow_html=True)
        st.sidebar.image(image_1, width=301)

#Deploy the model if the user uploads an image
if uploaded_image is not None:
    #Close the demo
    sel_img_path=0
    #Deploy the model with the uploaded image
    show_img(file_path=None, uploaded_image=uploaded_image, uploaded_state=True, demo_state=False)


#Deploy the model if the user selects Image 1
if sel_img_path== img_1_path:
    show_img(file_path=sel_img_path, uploaded_image=None, uploaded_state=False, demo_state=True)
    del uploaded_image


#Deploy the model if the user selects Image 2
if sel_img_path== img_2_path:
    show_img(file_path=sel_img_path, uploaded_image=None, uploaded_state=False, demo_state=True)
    del uploaded_image



#Deploy the model if the user selects Image 3
if sel_img_path== img_3_path:
    show_img(file_path=sel_img_path, uploaded_image=None, uploaded_state=False, demo_state=True)
    del uploaded_image

st.markdown('***')


if st.button('Classifiy (Predict)'):
    try:
        if uploaded_image is not None:
            deploy(file_path=None,uploaded_image=uploaded_image, uploaded_state=True, demo_state=False)
    except:
        deploy(sel_img_path)
    
    
    
if st.button('Explain'):
    try:
        if uploaded_image is not None:
            apply_gradCAM(img_path=None,uploaded_image=uploaded_image, uploaded_state=True, demo_state=False)
    except:
        apply_gradCAM(sel_img_path)
        
        
if st.button('Object Detect'):
    try:
        if uploaded_image is not None:
            bounding_box_prediction(model_OD, img_path=None, detection_threshold = decision_boundary, uploaded_image=uploaded_image, uploaded_state=True, demo_state=False)
    except:
        bounding_box_prediction(model_OD, sel_img_path, detection_threshold = decision_boundary)
