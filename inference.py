import torch
import os
from modules.tokenizers import Tokenizer
from models.models import XProNet
from modules.dataloaders import R2DataLoader
from modules.utils import parse_agrs
from PIL import Image
import json

def generate_report_for_image(args, image_id):
    # Load tokenizer
    tokenizer = Tokenizer(args)

    # Build model
    model = XProNet(args, tokenizer)
    model.load_state_dict(torch.load(args.resume, map_location='cpu'))  # pretrained weights
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create data loader only for the test split
    dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    dataset = dataloader.dataset

    # Lookup image
    idx = None
    for i, item in enumerate(dataset.entries):
        if item['image_id'] == image_id:
            idx = i
            break

    if idx is None:
        print(f"[ERROR] Image ID {image_id} not found in dataset.")
        return

    # Get the image and data
    sample = dataset[idx]
    image = sample['image'].unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output_ids = model.sample(image)
        report = tokenizer.decode(output_ids[0])

    print(f"\nGenerated Report for {image_id}:\n{report}\n")

if __name__ == "__main__":
    args = parse_agrs()
    args.batch_size = 1
    args.shuffle = False
    args.drop_last = False
    args.max_seq_length = 60  # or match your training
    args.dataset_name = 'iu_xray'

    # Must specify trained model path
    args.resume = 'checkpoints/iu_xray/xpronet_best_model.pth'
    args.split = 'test'  # needed for R2DataLoader

    # Set image ID (without `.png`)
    image_id = 'CXR1234_IM-00012-0001'

    generate_report_for_image(args, image_id)
