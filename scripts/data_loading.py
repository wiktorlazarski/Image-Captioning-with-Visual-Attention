import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision import datasets as dset

import scripts.data_processing as dp


class DatasetType(Enum):
    """Dataset types used in project"""

    TRAIN = 1
    VALIDATION = 2
    TEST = 3


@dataclass
class CocoTrainingDatasetPaths:
    """Paths to directories with data used for training Neural Net."""

    images: str
    captions_json: str


TRAINING_DATASET_PATHS: Dict[DatasetType, CocoTrainingDatasetPaths] = {
    DatasetType.TRAIN: CocoTrainingDatasetPaths(
        images="./data/train/train2017",
        captions_json="./data/train/captions_train2017.json",
    ),
    DatasetType.VALIDATION: CocoTrainingDatasetPaths(
        images="./data/validation/val2017",
        captions_json="./data/validation/captions_val2017.json",
    ),
}


class CocoCaptions(dset.VisionDataset):
    """Class provides easy access to preprocessed data."""

    def __init__(
        self,
        dset_paths: CocoTrainingDatasetPaths,
        transform: Optional[Callable] = None,
        target_transform: Optional[dp.TextPipeline] = None,
    ):
        """Constructor

        Args:
            dset_paths (CocoTrainingDatasetPaths): Paths to image folder and json with annotations
            transform (Optional[Callable], optional): Image preprocessing pipeline. Defaults to None.
            target_transform (Optional[Callable], optional): Text preprocessing pipeline. Defaults to None.
        """
        super().__init__(
            root=dset_paths.images, transform=transform, target_transform=target_transform
        )

        self.coco = COCO(dset_paths.captions_json)
        self.dataset = self._sort_dataset_on_token_count()

    def __getitem__(
        self, index: int
    ) -> Tuple[Union[Image.Image, torch.tensor], Union[str, List[int]]]:
        """Provide access to dataset via index.

        Args:
            index (int): Index of sample.

        Returns:
            Tuple[Union[Image.Image, torch.tensor], Union[str, torch.tensor]]: Preprocessed or not image and target caption.
        """
        image_id, caption = self.dataset[index]

        image = self._load_image(image_id)
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            caption = self.target_transform(caption)

        return image, caption

    def __len__(self) -> int:
        """Number of samples in dataset.

        Returns:
            int: Number of samples in dataset.
        """
        return len(self.dataset)

    def shuffle(self, subset_len: int) -> None:
        """Shuffle subset_len elements subsets of dataset.

        Args:
            subset_len (int): Shuffle subset length.
        """
        import random
        import math

        new_dataset_order = []
        for i in range(math.ceil((len(self.dataset) / subset_len))):
            subset = self.dataset[i * subset_len : (i + 1) * subset_len]
            new_dataset_order.extend(random.sample(subset, len(subset)))

        self.dataset = new_dataset_order

    def _sort_dataset_on_token_count(self) -> List[Tuple[int, str]]:
        """Function which reorganized dataset by sorting data by number of tokens in captions.
        Reorganization is done to increase training speed performance.

        Returns:
            List[Tuple[int, str]]: Image id and caption.
        """
        num_tokens_groups: Dict[int, List[Tuple[int, int]]] = {}

        for image_id in self.coco.imgs:
            annotations_id = self.coco.getAnnIds(image_id)
            annotations = self.coco.loadAnns(annotations_id)
            captions = [annotation["caption"] for annotation in annotations]

            for caption in captions:
                caption = caption.strip()
                normalized_caption = dp.TextPipeline.normalize(caption)

                token_count = len(normalized_caption.split())

                if token_count not in num_tokens_groups.keys():
                    num_tokens_groups[token_count] = []

                num_tokens_groups[token_count].append((image_id, caption))

        sorted_dataset = []
        for _, dataset in sorted(num_tokens_groups.items(), reverse=True):
            sorted_dataset.extend(dataset)

        return sorted_dataset

    def _load_image(self, image_id: int) -> Image.Image:
        """Load image from file.

        Args:
            image_id (int): Image id

        Returns:
            Image.Image: Image object
        """
        img_filename = self.coco.loadImgs(image_id)[0]["file_name"]
        img_path = os.path.join(self.root, img_filename)

        return Image.open(img_path).convert("RGB")


class CocoLoader(torch.utils.data.DataLoader):
    def __init__(self, coco_dataset: CocoCaptions, batch_size: int, num_workers: int):
        super().__init__(
            dataset=coco_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(
        self, batch: Tuple[torch.tensor, List[List[int]]]
    ) -> Tuple[torch.tensor, torch.LongTensor]:
        """Preprocess batch of images and caption before presenting to model.

        Args:
            batch (Tuple[torch.tensor, List[List[int]]]): Images and captions

        Returns:
            Tuple[torch.tensor, torch.tensor]: Preprocessed batch.
        """
        img_tensors = []
        captions = []
        for image, caption in batch:
            img_tensors.append(image)
            captions.append(caption)

        captions = self.dataset.target_transform.pad_sequences(captions)
        return torch.stack(img_tensors), torch.LongTensor(captions)
