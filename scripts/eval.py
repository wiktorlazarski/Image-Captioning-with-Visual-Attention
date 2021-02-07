import nltk.translate.bleu_score as bleu
import torch
import torchvision

import data_loading as dl
import data_processing as dp
import model


class CocoValidator:
    def __init__(self, dset_paths: dl.CocoTrainingDatasetPaths, vocabulary: dp.Vocabulary):
        self.coco_val = torchvision.datasets.CocoCaptions(dset_paths.images, dset_paths.captions_json, dp.VGGNET_PREPROCESSING_PIPELINE)
        self.vocabulary = vocabulary

    def validate(self, encoder: model.VGG19Encoder, decoder: model.LSTMDecoder, device: torch.device) -> float:
        """Computes average BLEU-4 score for validation dataset.

        Args:
            encoder (model.VGG19Encoder): image encoder
            decoder (model.LSTMDecoder): caption generator
            device (torch.device): train device

        Returns:
            float: average BLEU-4 score
        """
        SOS_INDEX = self.vocabulary.word2idx("<SOS>")
        EOS_INDEX = self.vocabulary.word2idx("<EOS>")

        bleu_sum = 0.0

        for image, captions in self.coco_val:
            preprocessed_captions = [dp.TextPipeline.normalize(caption).split() for caption in captions]

            max_length = max([len(caption) for caption in preprocessed_captions])

            image = image.to(device)

            feature_maps, feature_mean = encoder.forward(image.unsqueeze(0))

            sequence, _ = decoder.greedy_decoding(
                feature_maps=feature_maps,
                feature_mean=feature_mean,
                start_token_index=SOS_INDEX,
                end_token_index=EOS_INDEX,
                max_length=max_length,
            )

            sequence = dp.TextPipeline.decode_caption(self.vocabulary, sequence).split()

            bleu_4 = bleu.sentence_bleu(preprocessed_captions, sequence, (0.0, 0.0, 0.0, 1.0), bleu.SmoothingFunction().method1)

            bleu_sum += bleu_4

        return bleu_sum / len(self.coco_val)
