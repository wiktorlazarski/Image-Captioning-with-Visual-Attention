import base64
import os

from flask import Flask, jsonify, request
from scripts import model

from backend import utils

DECODER_PATH = os.environ["FLASK_APP_DECODER_PATH"]

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

    sequence, _ = decoder.greedy_decoding(
        feature_maps=feature_maps,
        feature_mean=feature_mean,
        start_token_index=utils.VOCABULARY.word2idx("<SOS>"),
        end_token_index=utils.VOCABULARY.word2idx("<EOS>"),
        max_length=100,
    )

    caption = utils.decode_caption(sequence)
    return caption
