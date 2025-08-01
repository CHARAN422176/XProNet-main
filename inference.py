import torch
import os
from models.models import XProNet
from modules.dataloaders import R2DataLoader
from modules.tokenizers import Tokenizer
from modules.utils import parse_agrs
import argparse


def get_args():
    # Step 1: parse original args
    args = parse_agrs()

    # Step 2: manually add new custom args
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_id', type=str, required=True, help="Image ID without .png")
    parser.add_argument('--resume', type=str, required=True, help="Path to trained model")

    # Step 3: parse only known args again (to override or extend)
    extra_args, _ = parser.parse_known_args()

    # Step 4: manually inject into args
    args.image_id = extra_args.image_id
    args.resume = extra_args.resume

    return args


def main():
    args = get_args()

    # Tokenizer and model
    tokenizer = Tokenizer(args)
    model = XProNet(args, tokenizer)
    model.load_state_dict(torch.load(args.resume, map_location='cpu'))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load test dataset
    dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    dataset = dataloader.dataset

    # Find index of the image
    image_idx = next((i for i, x in enumerate(dataset.entries) if x['image_id'] == args.image_id), None)

    if image_idx is None:
        print(f"[ERROR] Image ID '{args.image_id}' not found in test set.")
        return

    sample = dataset[image_idx]
    image_tensor = sample['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output_ids = model.sample(image_tensor)
        report = tokenizer.decode(output_ids[0])

    print(f"\nGenerated Report for {args.image_id}:\n{report}\n")

if __name__ == "__main__":
    main()
