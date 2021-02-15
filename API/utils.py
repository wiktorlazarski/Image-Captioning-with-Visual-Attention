import io
from typing import List

import scripts.data_processing as dp
import torch
from PIL import Image
from scripts import model

VOCABULARY = dp.Vocabulary()


def load_decoder(state_path: str) -> model.LSTMDecoder:
    decoder_state = torch.load(state_path, map_location=torch.device("cpu"))["decoder"]

    num_embeddings = decoder_state["word_embedding.weight"].shape[0]
    embedding_dim = decoder_state["word_embedding.weight"].shape[1]
    encoder_dim = 196
    attention_dim = decoder_state["attention.W_1.weight"].shape[0]
    decoder_dim = decoder_state["init_c.weight"].shape[0]

    decoder = model.LSTMDecoder(
        num_embeddings, embedding_dim, encoder_dim, decoder_dim, attention_dim
    )
    decoder.load_state_dict(decoder_state)

    return decoder


def preprocess_image(image_bytes: bytes) -> torch.tensor:
    image = Image.open(io.BytesIO(image_bytes))
    image = dp.VGGNET_PREPROCESSING_PIPELINE(image)

    return image.unsqueeze(0)


def decode_caption(decoded_seq: List[int]) -> str:
    caption = dp.TextPipeline.decode_caption(VOCABULARY, decoded_seq)
    caption = caption.capitalize() + "."

    return caption
