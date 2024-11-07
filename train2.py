import os

from accelerate import Accelerator
import datasets
import torch
import torch.distributed as dist
import wandb
from datasets import load_dataset
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from transformers import Blip2ForConditionalGeneration, Blip2Processor


def collate_fn(batch):
    batch = [b for b in batch if b["image"] is not None and b["text_input"] is not None]    
    processed_batch = {}

    images = [b["image"] for b in batch]
    processed_images = processor(images=images, return_tensors="pt")["pixel_values"]
    processed_batch["pixel_values"] = processed_images  # Shape should be [batch_size, 3, height, width]

    # Process text inputs using tokenizer
    text_inputs = processor.tokenizer(
        [example["text_input"] for example in batch],
        padding="max_length",
        max_length=40,
        return_tensors="pt",
        truncation=True,
    )
    processed_batch["input_ids"] = text_inputs["input_ids"]
    processed_batch["attention_mask"] = text_inputs["attention_mask"]

    return processed_batch


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

    wandb_project = "BLIP-2 Finetuning v3"
    run_name = "test"

    train_dataset_path = "/home/charles/VLM/data/onout_product_train_384_small.parquet"
    validation_dataset_path = "/home/charles/VLM/data/onout_product_validation_384_small.parquet"
    # train_dataset_path = "./onout_product_train_384_small.parquet"
    # validation_dataset_path = "./onout_product_validation_384_small.parquet"
    model_save_dir = "./ckpt/test"
    model_type = "pretrain_opt2.7b"

    batch_size = 8
    patience = 5
    num_epochs = 10
    learning_rate = 1e-5

    torch.manual_seed(23)

    ## init model
    checkpoint_path = "Salesforce/blip2-opt-2.7b"

    model = Blip2ForConditionalGeneration.from_pretrained(checkpoint_path,
        device_map = "auto",
        load_in_8bit = True,
        torch_dtype = torch.float16,
    )
    processor = Blip2Processor.from_pretrained(checkpoint_path)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    accelerator = Accelerator()
    model, optimizer, train_dataloader, valid_dataloader  = accelerator.prepare(model, optimizer, train_dataloader, validation_dataloader)
    device = accelerator.device

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

        # first_device = next(model.parameters()).device
        for step, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device).to(torch.float16)
            # pixel_values = batch.pop("pixel_values").to(first_device)

            outputs = model(
                input_ids = input_ids,
                pixel_values = pixel_values,
                labels = input_ids
            )

            loss = outputs.loss

            accelerator.backward(loss)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            total_loss += loss.item()

            if (step + 1) % 10 == 0:
                wandb.log({"train_loss": loss.item()})
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)

        total_loss_world = torch.tensor(total_loss, device="cuda")
        dist.all_reduce(total_loss_world, op=dist.ReduceOp.SUM)
        mean_loss = total_loss_world.item() / get_world_size() / len(train_dataloader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}, Mean Loss: {mean_loss}")
        wandb.log({"mean_train_loss": mean_loss})
        wandb.log({"train_loss": avg_loss})

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for idx, batch in enumerate(validation_dataloader):
                input_ids = batch.pop("input_ids")
                pixel_values = batch.pop("pixel_values").to(device).to(torch.float16)
                # pixel_values = batch.pop("pixel_values").to(first_device)

                outputs = model(
                    input_ids = input_ids,
                    pixel_values = pixel_values,
                    labels = input_ids
                )

                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(validation_dataloader)

        total_val_loss_world = torch.tensor(total_val_loss, device="cuda")
        dist.all_reduce(total_val_loss_world, op=dist.ReduceOp.SUM)
        mean_val_loss = total_val_loss_world.item() / get_world_size() / len(validation_dataloader)

        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Mean Validation Loss: {mean_val_loss:.4f}")
        wandb.log({"mean_validation_loss": mean_val_loss})
        wandb.log({"validation_loss": avg_val_loss})

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            epochs_no_improve = 0
            if dist.get_rank() == 0:  # Ensure only rank 0 saves the model
                model.save_pretrained(os.path.join(model_save_dir, f"epoch-{epoch+1}", f"{epoch+1}_blip2_model_v3.pth"))
            
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Stopping early at epoch {epoch + 1} due to no improvement.")
            break


    wandb.finish()
