import lhotse
import torch
import torchaudio

from dialogue_separator.model.lightning_module import DialogueSeparatorLightningModule

if __name__ == "__main__":
    ckpt_path = "dialogue-separator/k63bp3cm/checkpoints/epoch=699-step=63000.ckpt"
    dialogue_separator = DialogueSeparatorLightningModule.load_from_checkpoint(
        ckpt_path
    )

    urls = [
        "https://s3ds.mdx.jp/jchat/wds/youtube/test/data-00-000000.tar.gz",
        "https://s3ds.mdx.jp/jchat/wds/youtube/test/data-01-000000.tar.gz",
        "https://s3ds.mdx.jp/jchat/wds/youtube/test/data-02-000000.tar.gz",
    ]

    cutset = lhotse.CutSet.from_webdataset(urls)
    for cut in cutset.data:
        cut.save_audio("dialogue.wav")

        wav = torch.from_numpy(cut.load_audio())

        est1, est2 = dialogue_separator.separate_wav(wav, cut.sampling_rate)
        torchaudio.save("est1.wav", est1, 24000)
        torchaudio.save("est2.wav", est2, 24000)

        break
