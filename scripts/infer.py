from huggingface_hub import hf_hub_download

if __name__ == "__main__":
    ckpt_path = hf_hub_download(
        "koacai/dialogue-separator", "Libri2Mix/epoch=50-step=25449.ckpt"
    )
    print(ckpt_path)
