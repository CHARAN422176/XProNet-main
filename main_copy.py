import argparse

import numpy as np
import torch
import random
import torch.nn as nn
import torch.distributed as dist
import os

from models.models import XProNet
from modules.dataloaders import R2DataLoader
from modules.loss import compute_loss
from modules.metrics import CaptionScorer
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.tokenizers import Tokenizer
from modules.trainer import Trainer
from modules.utils import parse_agrs
from modules.logger import create_logger


def setup(world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = world_size


def main():
    # parse arguments
    args = parse_agrs()

    # DDP setup
    world_size = args.n_gpu

    torch.cuda.set_device(args.local_rank)
    if dist.is_available() and 'RANK' in os.environ:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size)
        rank = dist.get_rank()
    else:
        rank = 0
        world_size = 1
    device_id = rank % torch.cuda.device_count()

    # fix random seeds
    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    save_dir = os.path.join(args.output, args.dataset_name, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    logger = create_logger(output_dir=save_dir, dist_rank=args.local_rank, name=args.exp_name)

    # create tokenizer
    if args.dataset_name == 'cxr_gnome':
        tokenizer = None
    else:
        tokenizer = Tokenizer(args)

    # create data loaders
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True, drop_last=True)
    if args.dataset_name == 'cxr_gnome':
        tokenizer = train_dataloader.dataset.tokenizer
    all_texts = tokenizer.all_texts

    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model
    model = XProNet(args, tokenizer)
    optimizer = build_optimizer(args, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device_id)
    model.device = device
    model_without_ddp = model
    orig_topk = args.topk
    args.topk = 1


    # PARAMETER COUNT AND FLOPS COMPUTATION
    if rank == args.local_rank:
        logger.info(args)
        logger.info(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")

        # Parameters count
        n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")

        try:
            from ptflops import get_model_complexity_info

            class PtflopsWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                # Forward method matching your model's forward signature
                def forward(self, images, labels):
                    return self.model(images, labels=labels)

            wrapped_model = PtflopsWrapper(model_without_ddp).to("cpu").eval()

            # Settings based on your data loader info
            num_views = 2  # adjust if your model accesses higher index views
            num_labels = 14  # number of labels as per IU-XRay dataset

            input_shape = (num_views, 3, 224, 224)  # no batch dim; ptflops expects input shape excluding batch

            def input_constructor(input_res):
                batch_size = 1  # ptflops default batch size
                images = torch.randn(batch_size, *input_res)
                labels = torch.zeros(batch_size, num_labels, dtype=torch.float32)
                return {'images': images, 'labels': labels}

            with torch.no_grad():
                macs, params = get_model_complexity_info(
                    wrapped_model,
                    input_shape,
                    as_strings=True,
                    print_per_layer_stat=False,
                    verbose=False,
                    input_constructor=input_constructor,
                )

            logger.info(f"FLOPs: {macs}   Params: {params}")

            # Move model back to original device
            model_without_ddp.to(device_id)

        except Exception as e:
            logger.warning(f"Could not compute FLOPs due to error: {e}")

        if hasattr(model_without_ddp, 'flops'):
            try:
                flops = model_without_ddp.flops()
                logger.info(f"number of GFLOPs (model.flops()): {flops / 1e9}")
            except Exception as e:
                logger.warning(f"Error calling model.flops(): {e}")

    # # get function handles of loss and metrics
    # criterion = compute_loss
    # metrics = CaptionScorer(all_texts)

    # # build optimizer, learning rate scheduler
    # lr_scheduler = build_lr_scheduler(args, optimizer)

    # # build trainer and start training
    # trainer = Trainer(
    #     model,
    #     criterion,
    #     metrics,
    #     optimizer,
    #     args,
    #     lr_scheduler,
    #     logger,
    #     train_dataloader,
    #     val_dataloader,
    #     test_dataloader,
    # )
    # trainer.train()


if __name__ == '__main__':
    main()
