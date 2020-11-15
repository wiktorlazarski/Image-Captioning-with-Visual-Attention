import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import PIL
from pycocotools.coco import COCO
from torchvision import datasets as dset


@dataclass
class CoCoTrainingDatasetPaths:
    images: Union[str, Path]
    captions_json: Union[str, Path]
    segmentation_json: Union[str, Path]


COCO_TRAIN_DATASET_PATHS = CoCoTrainingDatasetPaths(
    images=Path("./data/validation/train2017"),
    captions_json=Path("./data/validation/captions_train2017.json"),
    segmentation_json=Path("./data/validation/instances_train2017.json"),
)

COCO_VALIDATION_DATASET_PATHS = CoCoTrainingDatasetPaths(
    images=Path("./data/validation/val2017"),
    captions_json=Path("./data/validation/captions_val2017.json"),
    segmentation_json=Path("./data/validation/instances_val2017.json"),
)


class CoCoTrainingDataset(dset.VisionDataset):
    Caption = str
    ObjectCategory = str
    SegmentationMask = List[float]

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
            if not isinstance(annotation["segmentation"], list):
                continue

            category_index = annotation["category_id"]
            category_name = self.coco_segmentation.loadCats(ids=category_index)[0]["name"]

            if category_name not in segmentations.keys():
                segmentations[category_name] = annotation["segmentation"]
            else:
                segmentations[category_name].append(annotation["segmentation"][0])

        return segmentations

    def _load_image(self, image_id: int) -> PIL.Image.Image:
        img_filename = self.coco_caption.loadImgs(image_id)[0]["file_name"]
        img_path = os.path.join(self.root, img_filename)

        return PIL.Image.open(img_path).convert("RGB")
