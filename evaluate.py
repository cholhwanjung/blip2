import argparse
from io import BytesIO

import pandas as pd
import torch
from evaluate import load
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from LAVIS.lavis.models import *
from LAVIS.lavis.models import (Blip2OPT, load_model,
                                load_model_and_preprocess, load_preprocess)
from LAVIS.lavis.processors import *


def chunk_images(image_list, batch_size):
    """Split the image list into chunks of a given batch size."""
    for i in range(0, len(image_list), batch_size):
        return image_list[i:i + batch_size]

def generate_captions_in_batches(images, vis_processor, model, device, batch_size):
    all_captions = []
    for batch in tqdm(chunk_images(images, batch_size)):
        # Preprocess the current batch
        processed_images = [vis_processor(Image.open(BytesIO(image))).unsqueeze(0).to(device) for image in batch]
        batch_images = torch.cat(processed_images, dim=0)
        
        # Generate captions for the batch
        with torch.no_grad():
            captions = model.generate({"image": batch_images})

        torch.cuda.empty_cache()
        
        all_captions.extend(captions)
    
    return all_captions

def main(args):
    batch_size = 16
    model_type = "pretrain_opt2.7b"
    checkpoint_path = "./ckpt/base/blip2_model.pth"
    test_data_path = "./facad_test_data.parquet"

    test_data = pd.read_parquet(test_data_path)
    image_bytes = list(test_data["image_bytes"])

    cfg = OmegaConf.load(Blip2OPT.default_config_path(model_type))
    from_checkpoint = True
    
    if from_checkpoint:
        model = load_model("blip2_opt", model_type, checkpoint=checkpoint_path)
        vis_processors, text_processors = load_preprocess(cfg["preprocess"])
    else:
        model, vis_processors, text_processors = load_model_and_preprocess(
            name="blip2_opt", model_type=model_type,
        )

    vis_processor = vis_processors["eval"]
    text_processor = text_processors["eval"]
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    captions = generate_captions_in_batches(image_bytes, vis_processor, model, device, batch_size=batch_size)

    bleu = load("bleu")
    rouge = load("rouge")
    meteor = load("meteor")

    references = list(test_data["true"])
    predictions = captions

    # BLEU-4
    bleu_score = bleu.compute(predictions=predictions, references=references)
    print("BLEU-4:", bleu_score["bleu"])

    # ROUGE-L
    rouge_score = rouge.compute(predictions=predictions, references=references)
    print("ROUGE-L:", rouge_score["rougeL"])

    # METEOR
    meteor_score = meteor.compute(predictions=predictions, references=references)
    print("METEOR:", meteor_score["meteor"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model with specific GPU")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device number to use")
    args = parser.parse_args()
    
    main(args)