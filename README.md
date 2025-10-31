# Geneses

- [Lightning AI](https://lightning.ai/koacai/speech/models/geneses)
- [Hugging Face](https://huggingface.co/koacai/geneses)
- [Evaluation metrics](https://github.com/urgent-challenge/urgent2025_challenge/tree/main/evaluation_metrics)

## Training

### Setup LibriTTS-R Mix

1. Fix the download directory in [`config/data/libritts_r_mix.yaml`](https://github.com/koacai/geneses/blob/main/config/data/libritts_r_mix.yaml)

<details>

<summary>`libritts_r_mix.yaml`</summary>

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

</details>

2. Run

```sh
uv run scripts/setup_libritts_r_mix.py
```

### Setup Noise & RIR Dataset for Test

1. Fix the download directory in [`config/data/libritts_r_mix.yaml`](https://github.com/koacai/geneses/blob/main/config/data/libritts_r_mix.yaml)

<details>

<summary>`libritts_r_mix.yaml`</summary>

```libritts_r_mix.yaml
test_noise_dataset:
  corpus:
    _target_: lhotse_dataset.DEMAND
  shard_dir: /groups/gag51394/users/asai/geneses/demand # NEED TO CHANGE
  shard_maxcount: 10

test_rir_dataset:
  corpus:
    _target_: lhotse_dataset.MITEnvironmentalImpulseResponses
  shard_dir: /groups/gag51394/users/asai/geneses/mit_environmental_impulse_responses # NEED TO CHANGE
  shard_maxcount: 10
```

</details>

2. Run

```sh
uv run scripts/setup_noise_rir.py
```

### Preprocess

1. Fix config in [`config/data/libritts_r_mix.yaml`](https://github.com/koacai/geneses/blob/main/config/data/libritts_r_mix.yaml)

<details>

<summary>`libritts_r_mix.yaml`</summary>

```libritts_r_mix.yaml
preprocess_datamodule:
  shard_dir: ${data.dataset.shard_dir}
  noise_dir: /groups/gcb50354/kohei_asai/dataset_stripe/noise/ # NEED TO CHANGE
  test_noise_dir: ${data.test_noise_dataset.shard_dir}
  test_rir_dir: ${data.test_rir_dataset.shard_dir}

  out_dir: /groups/gag51394/users/asai/geneses/libritts_r_mix_complex # NEED TO CHANGE

  only_bg: false # only background noise or complex degradation

  batch_size: 1
  num_workers: 32

  shard_maxcount:
    train: 1000
    valid: 30
    test: 30

  vae:
    hf_hub:
      repo_id: koacai/geneses
      filename: dacvae_l16_librittsr.pt
    hidden_size: 16
    max_duration: 20
    sample_rate: 24000
    noise_amp: 1e-4 # 0 paddingでは再構成できないため、微小なノイズを付与している

  ssl_model:
    name: facebook/w2v-bert-2.0
    sample_rate: 16000
```

</details>

2. Run

```sh
uv run scripts/preprocess.py # train and val dataset
uv run scripts/preprocess_test.py # test dataset
```
