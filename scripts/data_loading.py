import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision import datasets as dset

import scripts.data_preprocessing as dp


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
    ) -> Tuple[Union[Image.Image, torch.Tensor], Union[str, torch.Tensor]]:
        """Provide access to dataset via index.

        Args:
            index (int): Index of sample.

        Returns:
            Tuple[Union[Image.Image, torch.Tensor], Union[str, torch.Tensor]]: Preprocessed or not image and target caption.
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
                token_count = len(caption.split())

                if token_count not in num_tokens_groups.keys():
                    num_tokens_groups[token_count] = []

                num_tokens_groups[token_count].append((image_id, caption))

        sorted_dataset = []
        for _, dataset in sorted(num_tokens_groups.items(), reverse=True):
            sorted_dataset.extend(dataset)

        return sorted_dataset

    def _load_image(self, image_id: int) -> Image.Image:
        img_filename = self.coco.loadImgs(image_id)[0]["file_name"]
        img_path = os.path.join(self.root, img_filename)

        return Image.open(img_path).convert("RGB")
