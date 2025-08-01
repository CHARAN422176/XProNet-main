import torch
from models.models import XProNet
from modules.dataloaders import R2DataLoader
from modules.tokenizers import Tokenizer
from modules.utils import parse_agrs

# === Set your image ID and checkpoint ===
image_id = 'CXR49_IM-2110'
resume_path = 'results/iu_xray/model_best.pth'  # Update this if needed

# === Load args, model, tokenizer ===
args = parse_agrs()
args.batch_size = 1
args.shuffle = False
args.drop_last = False
args.split = 'test'
args.resume = resume_path
args.beam_size = 3
args.dataset_name = 'iu_xray'

tokenizer = Tokenizer(args)
model = XProNet(args, tokenizer)
model.load_state_dict(torch.load(args.resume, map_location='cpu'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === Load test data ===
dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
dataset = dataloader.dataset

# === Get image by ID ===
image_idx = next((i for i, entry in enumerate(dataset.entries) if entry['image_id'] == image_id), None)

if image_idx is None:
    print(f"[ERROR] Image ID '{image_id}' not found.")
else:
    sample = dataset[image_idx]
    image_tensor = sample['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output_ids = model.sample(image_tensor)
        report = tokenizer.decode(output_ids[0])

    print(f"\n📋 Generated Report for image {image_id}:\n{report}\n")
