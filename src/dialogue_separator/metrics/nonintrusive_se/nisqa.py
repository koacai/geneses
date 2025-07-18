from dialogue_separator.metrics.nonintrusive_se.nisqa_util import load_nisqa_model


def calc_nisqa(use_gpu: bool) -> None:
    nisqa_model_path = "src/dialogue_separator/lib/NISQA/weights/nisqa.tar"
    model = load_nisqa_model(nisqa_model_path, device="cuda" if use_gpu else "cpu")
    print(model)
