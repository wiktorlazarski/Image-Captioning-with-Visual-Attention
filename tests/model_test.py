from typing import Tuple

import pytest
import scripts.data_loading as dl
import scripts.data_preprocessing as dp
import scripts.model as mdl
import torch


@pytest.fixture
def train_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    coco_dset = dl.CocoCaptions(
        dl.TRAINING_DATASET_PATHS[dl.DatasetType.VALIDATION],
        transform=dp.VGGNET_PREPROCESSING_PIPELINE,
        target_transform=dp.TextPipeline(),
    )

    coco_loader = dl.CocoLoader(coco_dset, batch_size=8, num_workers=2)
    return next(iter(coco_loader))


def test_cuda_availability() -> None:
    assert torch.cuda.is_available()


def test_vgg_encoder_output_dim(train_batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
    # given
    image_batch, _ = train_batch
    batch_size = len(image_batch)

    encoder = mdl.VGG19Encoder()
    expected_feature_maps_dim = torch.Size((batch_size, 512, 196))
    expected_feature_mean_dim = torch.Size((batch_size, 196))

    # when
    feature_maps, feature_mean = encoder.forward(image_batch)

    # then
    assert feature_maps.shape == expected_feature_maps_dim
    assert feature_mean.shape == expected_feature_mean_dim


def test_decoder_word_embeddings(train_batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
    # given
    _, caption_batch = train_batch
    batch_size = len(caption_batch)

    embedding_dim = 16
    decoder_word_embedding = mdl.LSTMDecoder(
        num_embeddings=10_004,
        embedding_dim=embedding_dim,
        encoder_dim=10,
        decoder_dim=10,
        attention_dim=10,
    ).word_embedding

    caption_len = len(caption_batch[0])
    expected_word_embedding_out_dim = torch.Size((batch_size, caption_len, embedding_dim))

    # when
    embeddings = decoder_word_embedding(caption_batch)

    # then
    assert embeddings.shape == expected_word_embedding_out_dim


def test_decoder_output_dim(train_batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
    # given
    image_batch, caption_batch = train_batch
    batch_size = len(caption_batch)
    caption_len = len(caption_batch[0])

    encoder = mdl.VGG19Encoder()
    decoder = mdl.LSTMDecoder(
        num_embeddings=10_004, embedding_dim=2, encoder_dim=196, decoder_dim=4, attention_dim=2
    )

    expected_prediction_dim = torch.Size((caption_len - 1, batch_size, 10_004))
    expected_context_dim = torch.Size((caption_len - 1, batch_size, 196))

    # when
    with torch.no_grad():
        predictions, contexts = decoder.forward(*encoder.forward(image_batch), caption_batch)

        # then
        assert predictions.shape == expected_prediction_dim
        assert contexts.shape == expected_context_dim


def test_attention_output_dim(train_batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
    # given
    image_batch, _ = train_batch
    feature_maps, _ = mdl.VGG19Encoder().forward(image_batch)

    batch_size = len(image_batch)
    mock_h = torch.rand((batch_size, 16))

    feature_dim = feature_maps.shape[-1]
    attention = mdl.AdditiveAttention(attention_dim=10, values_dim=feature_dim, query_dim=16)

    expected_attention_dim = torch.Size((batch_size, feature_dim))

    # when
    context = attention.forward(feature_maps, mock_h)

    # then
    assert context.shape == expected_attention_dim
