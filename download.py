from transformers import Blip2ForConditionalGeneration, Blip2Processor


if __name__ == "__main__":
    checkpoint_path = "Salesforce/blip2-opt-2.7b"
    save_dir = "./ckpt/base/opt-2.7b"

    model = Blip2ForConditionalGeneration.from_pretrained(checkpoint_path)
    processor = Blip2Processor.from_pretrained(checkpoint_path)

    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)