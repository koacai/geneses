# Geneses

- [Lightning AI](https://lightning.ai/koacai/speech/models/geneses)
- [Hugging Face](https://huggingface.co/koacai/geneses)
- [Evaluation metrics](https://github.com/urgent-challenge/urgent2025_challenge/tree/main/evaluation_metrics)

## Training

### Setup LibriTTS-R Mix

1. Fix the download directory in `[config/data/libritts_r_mix.yaml](https://github.com/koacai/geneses/blob/main/config/data/libritts_r_mix.yaml)`

```libritts_r_mix.yaml
dataset:
  libritts_r_shar_dir: /groups/gcb50354/kohei_asai/shar/libritts_r # NEED TO CHANGE

  libritts_r_mix:
    _target_: lhotse_dataset.LibriTTSRMixLarge
    libritts_r_shar_dir: ${data.dataset.libritts_r_shar_dir},
    num_test_clean: 5000
    num_dev_clean: 5000
    num_train_clean_100: 100000
    num_train_clean_360: 300000

  shard_dir: /groups/gag51394/users/asai/geneses/libritts_r_mix_raw # NEED TO CHANGE

  shard_maxcount:
    train: 1000
    valid: 30
    test: 30
```

2. Run

```sh
uv run scripts/setup_libritts_r_mix.py
```
