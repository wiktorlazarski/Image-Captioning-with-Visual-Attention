from dataclasses import dataclass
from enum import Enum
from typing import Dict

from torchvision import datasets as dset
from torchvision import transforms


class DatasetType(Enum):
    Train = 1
    Validation = 2
    Test = 3


@dataclass
class CocoTrainingDatasetPaths:
    images: str
    captions_json: str


TRAINING_DATASET_PATHS: Dict[DatasetType, CocoTrainingDatasetPaths] = {
    DatasetType.Train: CocoTrainingDatasetPaths(
        images="./data/validation/train2017",
        captions_json="./data/validation/captions_train2017.json",
    ),
    DatasetType.Validation: CocoTrainingDatasetPaths(
        images="./data/validation/val2017",
        captions_json="./data/validation/captions_val2017.json",
    ),
}

VGGNET_PREPROCESSING_PIPELINE = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_dataset(
    dataset_type: DatasetType = DatasetType.Train,
    vgg_preprocessed: bool = False,
    text_preprocessed: bool = False,
) -> dset.CocoCaptions:
    image_pipeline = VGGNET_PREPROCESSING_PIPELINE if vgg_preprocessed else None
    text_pipeline = None if text_preprocessed else None

    dataset_paths = TRAINING_DATASET_PATHS[dataset_type]

    return dset.CocoCaptions(
        root=dataset_paths.images,
        annFile=dataset_paths.captions_json,
        transform=image_pipeline,
        target_transform=text_pipeline,
    )
