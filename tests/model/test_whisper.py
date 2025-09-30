from faster_whisper import WhisperModel


def test_whisper() -> None:
    path = "test_output/0/0/wav_1.wav"
    whisper = WhisperModel("large-v3", device="cuda", compute_type="float16")
    segments, _ = whisper.transcribe(path)

    text = ""
    for segment in segments:
        text += segment.text

    print(text)
