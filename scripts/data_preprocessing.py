import string
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torchvision import transforms

VGGNET_PREPROCESSING_PIPELINE = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class Vocabulary:
    """Vocabulary class which maps tokens' indexes and words"""

    COCO_CAPTION_TOP10K_WORDS = "./resources/text/vocabulary_10k.csv"
    SPECIAL_TOKENS = ["<SOS>", "<EOS>", "<UNK>", "<PAD>"]

    def __init__(self, path: str = COCO_CAPTION_TOP10K_WORDS):
        """Constructor

        Args:
            path (str): Path to csv file containing 'WORD' column from which words will be loaded.
                        Defaults to COCO_CAPTION_TOP10K_WORDS.
        """
        most_frequent_words = pd.read_csv(path)["WORD"]
        self._idx2word, self._word2idx = self._create_mappings(most_frequent_words)

    def __len__(self) -> int:
        """Number of words in vocabulary.

        Returns:
            int: Number of words in vocabulary.
        """
        return len(self._idx2word)

    def word2idx(self, word: str) -> int:
        """Returns index of a given word.

        Args:
            word (str): Word

        Returns:
            int: Index of a given word. If word doesn't have mapping index of '<UNK>' is returned.
        """
        if word not in self._word2idx:
            return self._word2idx["<UNK>"]
        return self._word2idx[word]

    def idx2word(self, index: int) -> str:
        """Returns word of a given index.

        Args:
            index (int): Index

        Raises:
            IndexError: Raised when index does not have corresponding mapping in vocabulary.

        Returns:
            str: Word of a given index.
        """
        if index not in self._idx2word:
            raise IndexError(f"No word is mapped to {index}")
        return self._idx2word[index]

    def _create_mappings(self, words: pd.Series) -> Tuple[Dict[int, str], Dict[str, int]]:
        """Create two dictionaries used to efficiently return mappings.

        Args:
            words (pd.Series): Loaded from csv file words from column 'WORD'.

        Returns:
            Tuple[Dict[int, str], Dict[str, int]]: Mappings between indexes and words.
        """
        idx2word = words.to_dict()

        num_words = len(idx2word)
        for index, special_token in enumerate(self.SPECIAL_TOKENS):
            idx2word[num_words + index] = special_token

        word2idx = dict(zip(idx2word.values(), idx2word.keys()))

        return idx2word, word2idx


class TextPipeline:
    """Text pipeline used by Dataset class in order to preprocess text"""

    def __init__(self):
        self.vocabulary = Vocabulary()

    def __call__(self, text: str) -> List[int]:
        """Preprocess text by one-hot encoding every token.
        Moreover, add encoded '<SOS>'/'<EOS>' at the start/end of text.

        Args:
            text (str): Input text

        Returns:
            List[int]: Encoded text
        """
        tokens = self._normalize(text)

        encoded_tokens = [self.vocabulary.word2idx("<SOS>")]

        for token in tokens:
            token_index = self.vocabulary.word2idx(token)
            encoded_tokens.append(token_index)

        encoded_tokens.append(self.vocabulary.word2idx("<EOS>"))

        return encoded_tokens

    def pad_sequences(self, sequences: List[List[int]]) -> List[List[int]]:
        """Pad list of encoded captions.

        Args:
            sequences (List[List[int]]): List of encoded captions.

        Returns:
            List[List[int]]: Padded target values to the same length as first sample.
        """
        captions_batch_len = len(sequences[0])

        targets = [sequences[0]]
        for sequence in sequences[1:]:
            if len(sequence) == captions_batch_len:
                targets.append(sequence)
                continue

            num_paddings = captions_batch_len - len(sequence)
            sequence.extend([self.vocabulary.word2idx("<PAD>")] * num_paddings)

            targets.append(sequence)

        return targets

    def _normalize(self, text: str) -> List[str]:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation.replace("-", "")))

        return text.split()
