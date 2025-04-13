import torch
from cs336_basics.transformer import Transformer
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train import Config, load_config
from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.decoding import decode


def main():
    # vocab_path = "./out/tokenizers/ts-train/vocab.txt"
    # merges_path = "./out/tokenizers/ts-train/merges.txt"

    vocab_path = "./out/tokenizers/owt-train/vocab.txt"
    merges_path = "./out/tokenizers/owt-train/merges.txt"

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

    config = Config(load_config("./cs336_basics/configs/owt.yml"))
    model = Transformer(**config.model, device="cuda", dtype=torch.float32)
    model.to("cuda")

    # print("Pre-checkpoint")
    # print(decode(model, tokenizer, "The", max_new_tokens=512, temperature=0.7, top_p=0.9))

    # checkpoint = "../out/runs/latest/checkpoints/latest.pt"
    checkpoint = "/data/c-sniderb/runs/latest/checkpoints/latest.pt"
    # checkpoint = "/data/c-sniderb/runs/run-1744519021/checkpoints/latest.pt"
    load_checkpoint(checkpoint, model)

    # print("-" * 100)

    # print("Post-checkpoint")
    print(decode(model, tokenizer, "The", max_new_tokens=512, temperature=0.7, top_p=0.9))


if __name__ == "__main__":
    main()
