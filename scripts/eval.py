import nltk.translate.bleu_score as bleu
import torch
import torchvision

import scripts.data_loading as dl
import scripts.data_processing as dp
from scripts import model


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

        refs=[]
        hyps=[]
        for i, (image, captions) in enumerate(self.coco_val):
            preprocessed_captions = [dp.TextPipeline.normalize(caption).split() for caption in captions]

            max_length = max([len(caption) for caption in preprocessed_captions])

            image = image.to(device)

            feature_maps, feature_mean = encoder.forward(image.unsqueeze(0))

            sequence, _, _ = decoder.greedy_decoding(
                feature_maps=feature_maps,
                feature_mean=feature_mean,
                start_token_index=SOS_INDEX,
                end_token_index=EOS_INDEX,
                max_length=max_length,
            )

            sequence = dp.TextPipeline.decode_caption(self.vocabulary, sequence).split()
            hyps.append(sequence)
            refs.append(preprocessed_captions)

        return bleu.corpus_bleu(refs, hyps)


    def validate_beam(self, encoder: model.VGG19Encoder, decoder: model.LSTMDecoder, beam_size: int, num_sequences: int, device: torch.device) -> float:
        """Computes average BLEU-4 score for validation dataset.

        Args:
            encoder (model.VGG19Encoder): image encoder
            decoder (model.LSTMDecoder): caption generator
            beam_size (int): beam size
            num_sequences (int): number of sequences considered
            device (torch.device): train device

        Returns:
            float: average BLEU-4 score
        """
        SOS_INDEX = self.vocabulary.word2idx("<SOS>")
        EOS_INDEX = self.vocabulary.word2idx("<EOS>")

        refs=[]
        hyps=[]
        for image, captions in self.coco_val:
            preprocessed_captions = [dp.TextPipeline.normalize(caption).split() for caption in captions]

            max_length = max([len(caption) for caption in preprocessed_captions])

            image = image.to(device)

            feature_maps, feature_mean = encoder.forward(image.unsqueeze(0))

            sequences = decoder.beam_search(
                feature_maps=feature_maps,
                feature_mean=feature_mean,
                start_token_index=SOS_INDEX,
                end_token_index=EOS_INDEX,
                beam_size=beam_size,
                num_sequences=num_sequences,
                max_length=max_length,
            )
            sequences = sorted(sequences, reverse=True, key=lambda x: x[1])

            sequence = dp.TextPipeline.decode_caption(self.vocabulary, sequences[0][0]).split()
            hyps.append(sequence)
            refs.append(preprocessed_captions)

        return bleu.corpus_bleu(refs, hyps)