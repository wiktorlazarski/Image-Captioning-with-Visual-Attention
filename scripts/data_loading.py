import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision import datasets as dset


class DatasetType(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


@dataclass
class CocoTrainingDatasetPaths:
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
    def __init__(
        self,
        dset_paths: CocoTrainingDatasetPaths,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(
            root=dset_paths.images, transform=transform, target_transform=target_transform
        )

        self.coco = COCO(dset_paths.captions_json)
        self.dataset = self._sort_dataset_on_token_count()

    def __getitem__(
        self, index: int
    ) -> Tuple[Union[Image.Image, torch.Tensor], Union[str, torch.Tensor]]:
        image_id, caption = self.dataset[index]

        image = self._load_image(image_id)
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            caption = self.target_transform(caption)

        return image, caption

    def __len__(self) -> int:
        return len(self.dataset)

    def _sort_dataset_on_token_count(self) -> List[Tuple[int, str]]:
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
