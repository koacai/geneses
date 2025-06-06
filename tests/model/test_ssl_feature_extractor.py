from dialogue_separator.model.lightning_module import SSLFeatureExtractor


def test_ssl_feature_extractor() -> None:
    ssl_feature_extractor = SSLFeatureExtractor("facebook/w2v-bert-2.0", 13, True)
    assert hasattr(ssl_feature_extractor.model, "adapter")
