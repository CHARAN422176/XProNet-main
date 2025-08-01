import torch
from models.models import XProNet
from modules.dataloaders import R2DataLoader
from modules.tokenizers import Tokenizer
from modules.utils import parse_agrs

# === USER INPUT ===
image_id = 'CXR1234_IM-00012-0001'  # <-- change this to your image ID (without .png)
resume_path = 'results/iu_xray/model_best.pth'  # <-- change to your checkpoint path

# === Load args from training ===
args = parse_agrs()
args.batch_size = 1
args.shuffle = False
args.drop_last = False
args.split = 'test'
args.resume = resume_path
args.beam_size = 3  # or as used in training
args.dataset_name = 'iu_xray'

# === Build tokenizer and model ===
tokenizer = Tokenizer(args)
model = XProNet(args, tokenizer)
model.load_state_dict(torch.load(args.resume, map_location='cpu'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === Load test dataset ===
dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
dataset = dataloader.dataset

# === Find image by ID ===
image_idx = next((i for i, entry in enumerate(dataset.entries) if entry['image_id'] == image_id), None)

if image_idx is None:
    print(f"[ERROR] Image ID '{image_id}' not found in test split.")
else:
    sample = dataset[image_idx]
    image_tensor = sample['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output_ids = model.sample(image_tensor)
        report = tokenizer.decode(output_ids[0])

    print(f"\n📋 Report for image {image_id}:\n{report}\n")
