import os

import datasets
import torch
import torch.distributed as dist
import wandb
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from omegaconf import OmegaConf
from LAVIS.lavis.models import Blip2Qformer, load_model, load_preprocess, load_model_and_preprocess
from LAVIS.lavis.models import *
from LAVIS.lavis.processors import *

from src.stage_train import forward_stage1,forward_stage2


def collate_fn(batch):
    batch = [b for b in batch if b["image"] is not None and b["text_input"] is not None]
    images = torch.stack([vis_processor(image) for image in (b["image"] for b in batch)])
    # images = torch.stack([vis_processor(resize_transform(image)) for image in (b["image"] for b in batch)])
    
    # Collect text labels
    labels = [text_processor(b["text_input"]) for b in batch]
    
    return {"image": images, "text_input": labels}

def setup_ddp():
    """Initialize the DDP environment."""
    dist.init_process_group(backend="nccl")  # Use nccl for GPU, gloo for CPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """Clean up the DDP environment."""
    dist.destroy_process_group()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


if __name__ == "__main__":

    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_TIMEOUT"] = "1200"

    wandb_project = "BLIP-2 Finetuning"
    run_name = "test"

    train_dataset_path = "./onout_product_train_384_small.parquet"
    validation_dataset_path = "./onout_product_validation_384_small.parquet"
    model_save_dir = "./ckpt/test"
    model_type = "pretrain_opt2.7b"

    batch_size = 8
    patience = 5
    num_epochs = 10
    learning_rate = 1e-5

    local_rank = setup_ddp()
    torch.manual_seed(23)

    ## init model
    cfg = OmegaConf.load(Blip2OPT.default_config_path(model_type))
    from_checkpoint = True
    checkpoint_path = "./ckpt/base/blip2_model.pth"

    if local_rank == 0:
        if from_checkpoint:
            model = load_model("blip2_opt", model_type, checkpoint=checkpoint_path)
            vis_processors, text_processors = load_preprocess(cfg["preprocess"])
        else:
            model, vis_processors, text_processors = load_model_and_preprocess(
                name="blip2_opt", model_type=model_type,
            )
    dist.barrier()

    if local_rank != 0:
        if from_checkpoint:
            model = load_model("blip2_opt", model_type, checkpoint=checkpoint_path)
            vis_processors, text_processors = load_preprocess(cfg["preprocess"])
        else:
            model, vis_processors, text_processors = load_model_and_preprocess(
                name="blip2_opt", model_type=model_type,
            )

    vis_processor = vis_processors["eval"]
    text_processor = text_processors["eval"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    ## init dataset
    ds = load_dataset(
        "parquet",
        data_files={
            "train": train_dataset_path,
            "val": validation_dataset_path
        }
    )
    ds = ds.cast_column("image_bytes", datasets.features.Image())
    
    train_dataset, validation_dataset = ds["train"], ds["val"]
    
    train_dataset = train_dataset.rename_column("image_bytes", "image").rename_column("text", "text_input")
    validation_dataset = validation_dataset.rename_column("image_bytes", "image").rename_column("text", "text_input")

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn)

    validation_sampler = DistributedSampler(validation_dataset, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    forward_fn = forward_stage2

    best_val_loss = float("inf")
    epochs_no_improve = 0

    wandb.init(
        # set the wandb project where this run will be logged
        project=wandb_project,
        name=run_name,

        # track hyperparameters and run metadata
        config={
            "model_name": model_type,
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "model_save_dir": model_save_dir,
            "train_dataset_path": train_dataset_path,
        },
        group="DDP",
    )


    # Fine-tuning Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_sampler.set_epoch(epoch)
        validation_sampler.set_epoch(epoch)

        print(f"Start Training Epoch {epoch} @ {local_rank}")
        for step, batch in enumerate(train_dataloader):
            outputs = forward_fn(model.module, batch, local_rank)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (step + 1) % 100 == 0:
                wandb.log({"train_loss": loss.item()})
                print(f"Epoch [{epoch+1}/{num_epochs}], Rank [{local_rank}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)

        total_loss_world = torch.tensor(total_loss, device="cuda")
        dist.all_reduce(total_loss_world, op=dist.ReduceOp.SUM)
        mean_loss = total_loss_world.item() / get_world_size() / len(train_dataloader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Rank [{local_rank}], Loss: {avg_loss}, Mean Loss: {mean_loss}")
        wandb.log({"mean_train_loss": mean_loss})
        wandb.log({"train_loss": avg_loss})

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for idx, batch in enumerate(validation_dataloader):
                outputs = forward_fn(model.module, batch, local_rank)
                torch.cuda.synchronize()

                loss = outputs["loss"]
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(validation_dataloader)

        total_val_loss_world = torch.tensor(total_val_loss, device="cuda")
        dist.all_reduce(total_val_loss_world, op=dist.ReduceOp.SUM)
        mean_val_loss = total_val_loss_world.item() / get_world_size() / len(validation_dataloader)

        print(f"Epoch {epoch + 1}, Rank [{local_rank}], Validation Loss: {avg_val_loss:.4f}, Mean Validation Loss: {mean_val_loss:.4f}")
        wandb.log({"mean_validation_loss": mean_val_loss})
        wandb.log({"validation_loss": avg_val_loss})

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            epochs_no_improve = 0
            if dist.get_rank() == 0:  # Ensure only rank 0 saves the model
                os.makedirs(os.path.join(model_save_dir, f"epoch-{epoch+1}"), exist_ok = True)
                torch.save(model.state_dict(), os.path.join(model_save_dir, f"epoch-{epoch+1}", f"{epoch+1}_blip2_model.pth"))
            
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Stopping early at epoch {epoch + 1} due to no improvement.")
            break


    wandb.finish()
    cleanup_ddp()
