import base64
import os

from flask import Flask, jsonify, request
from scripts import model

from API import utils

DECODER_PATH = "./API/decoder_brain.pth"

app = Flask(__name__)

encoder = model.VGG19Encoder()
decoder = utils.load_decoder(DECODER_PATH)


@app.route("/caption", methods=["POST"])
def caption_image() -> None:
    if request.method == "POST":
        json = request.get_json()
        image_str = json["image"]

        image_caption = None
        try:
            image_bytes = base64.b64decode(image_str)
            image_caption = predict_caption(image_bytes)
        except Exception as e:
            return jsonify({"caption": f"ERROR {str(e)}"})

        return jsonify({"caption": image_caption})


def predict_caption(image_bytes: bytes) -> str:
    input_image = utils.preprocess_image(image_bytes)

    feature_maps, feature_mean = encoder.forward(input_image)

    sequences = decoder.beam_search(
        feature_maps=feature_maps,
        feature_mean=feature_mean,
        start_token_index=utils.VOCABULARY.word2idx("<SOS>"),
        end_token_index=utils.VOCABULARY.word2idx("<EOS>"),
        beam_size=3,
        num_sequences=1,
        max_length=100,
    )
    sequences = sorted(sequences, reverse=True, key=lambda x: x[1])

    caption = utils.decode_caption(sequences[0][0])
    return caption
