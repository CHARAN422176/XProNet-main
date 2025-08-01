import argparse
import sys
sys.path.append("/kaggle/working/XProNet-main")  # or wherever your project root is
import torch
from torchvision import transforms
from models.models import XProNet
from modules.tokenizers import Tokenizer
from modules.datasets import IuxrayMultiImageDataset
import json
import pickle
from PIL import Image
import os

# ======== SET THESE ========
IMAGE_ID = "CXR49_IM-2110"  # Image folder name (without extension)
IMAGE_FOLDER = "/kaggle/input/iu-xray/iu_xray/images"
ANNOTATION_PATH = "/kaggle/input/iu-xray/iu_xray/annotation.json"
LABEL_PATH = "files/iu_xray/labels/labels_14.pickle"
MODEL_PATH = "/kaggle/input/xpronet_iu_xray/pytorch/default/1/iu_xray.pth"  # Path to trained model
INIT_PROTOTYPES_PATH = "files/iu_xray/init_prototypes.pt"

# ======== PREPARE ARGUMENTS ========
class Args:
    ann_path = ANNOTATION_PATH
    image_dir = IMAGE_FOLDER
    label_path = LABEL_PATH
    init_protypes_path = INIT_PROTOTYPES_PATH
    dataset_name = "iu_xray"
    max_seq_length = 60
    num_cluster = 14
    num_protype = 20
    cmm_size = 2048
    cmm_dim = 512
    d_img_ebd = 2048
    d_txt_ebd = 768
    topk = 15
    beam_size = 3

args = Args()

# ======== SETUP =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ======== LOAD TOKENIZER AND MODEL =========
tokenizer = Tokenizer(args)
model = XProNet(args, tokenizer, mode='sample').to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ======== LOAD IMAGE PAIR =========
image_paths = [f"{IMAGE_ID}/0.png", f"{IMAGE_ID}/1.png"]
image_1 = transform(Image.open(os.path.join(IMAGE_FOLDER, image_paths[0])).convert('RGB'))
image_2 = transform(Image.open(os.path.join(IMAGE_FOLDER, image_paths[1])).convert('RGB'))
images = torch.stack((image_1, image_2), dim=0).unsqueeze(0).to(device)

# ======== LOAD LABEL =========
with open(LABEL_PATH, "rb") as f:
    labels = pickle.load(f)
array = IMAGE_ID.split('-')
mod_id = f"{array[0]}-{array[1]}"
label = torch.FloatTensor(labels[mod_id]).unsqueeze(0).to(device)

# ======== INFERENCE =========
with torch.no_grad():
    output, _ = model(images, labels=label, mode='sample')
    decoded_report = tokenizer.decode(output[0].cpu().numpy())

print("Generated Report:\n", decoded_report)
