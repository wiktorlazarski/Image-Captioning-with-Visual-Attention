import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import PIL
import numpy as np
from pycocotools.coco import COCO
from torchvision import datasets as dset


@dataclass
class CoCoTrainingDatasetPaths:
    images: str
    captions_json: str
    segmentation_json: str


COCO_TRAIN_DATASET_PATHS = CoCoTrainingDatasetPaths(
    images="./data/validation/train2017",
    captions_json="./data/validation/captions_train2017.json",
    segmentation_json="./data/validation/instances_train2017.json",
)

COCO_VALIDATION_DATASET_PATHS = CoCoTrainingDatasetPaths(
    images="./data/validation/val2017",
    captions_json="./data/validation/captions_val2017.json",
    segmentation_json="./data/validation/instances_val2017.json",
)


class CoCoTrainingDataset(dset.VisionDataset):
    Caption = str
    ObjectCategory = str
    SegmentationMask = np.array

    def __init__(
        self,
        data_paths: CoCoTrainingDatasetPaths,
        image_transform: Optional[Callable[[PIL.Image.Image], PIL.Image.Image]] = None,
        caption_transform: Optional[Callable[[List[Caption]], List[Caption]]] = None,
    ):
        super(CoCoTrainingDataset, self).__init__(
            root=data_paths.images, transform=image_transform, target_transform=caption_transform
        )
        self.coco_caption = COCO(data_paths.captions_json)
        self.coco_segmentation = COCO(data_paths.segmentation_json)
        self.image_ids = list(sorted(self.coco_caption.imgs.keys()))

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(
        self, index: int
    ) -> Tuple[PIL.Image.Image, List[Caption], Dict[ObjectCategory, List[SegmentationMask]]]:
        img_id = self.image_ids[index]
        captions = self._load_captions(img_id)

        if self.target_transform is not None:
            captions = self.target_transform(captions)

        segmentations = self._load_segmentations(img_id)
        image = self._load_image(img_id)

        if self.transform is not None:
            image = self.transform(image)

        return image, captions, segmentations

    def _load_captions(self, image_id: int) -> List[Caption]:
        annotations = self.coco_caption.getAnnIds(imgIds=image_id)
        captions = [annotation["caption"] for annotation in self.coco_caption.loadAnns(annotations)]

        return captions

    def _load_segmentations(self, image_id: int) -> Dict[ObjectCategory, List[SegmentationMask]]:
        annotations_ids = self.coco_segmentation.getAnnIds(imgIds=image_id)
        annotations = self.coco_segmentation.loadAnns(annotations_ids)

        segmentations = {}
        for annotation in annotations:
            category_index = annotation["category_id"]
            category_name = self.coco_segmentation.loadCats(ids=category_index)[0]["name"]
            category_name = category_name.strip()

            try:
                segment_mask = self.coco_segmentation.annToMask(annotation)
                if len(segment_mask) == 0:
                    continue

                if category_name not in segmentations:
                    segmentations[category_name] = []

                segmentations[category_name].append(segment_mask)
            except:
                pass

        return segmentations

    def _load_image(self, image_id: int) -> PIL.Image.Image:
        img_filename = self.coco_caption.loadImgs(image_id)[0]["file_name"]
        img_path = os.path.join(self.root, img_filename)

        return PIL.Image.open(img_path).convert("RGB")
