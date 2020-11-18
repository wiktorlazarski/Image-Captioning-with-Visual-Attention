import numpy as np
import pytest
import scripts.data_utils.data_loader as service
from PIL import Image


@pytest.fixture
def coco_dataset() -> service.CoCoTrainingDataset:
    return service.CoCoTrainingDataset(service.COCO_VALIDATION_DATASET_PATHS)


def test_len(coco_dataset: service.CoCoTrainingDataset) -> None:
    # given
    expected_result = 5_000

    # when
    result = len(coco_dataset)

    # then
    assert result == expected_result


def test_getitem_with_invalid_index(coco_dataset: service.CoCoTrainingDataset) -> None:
    # given
    index = len(coco_dataset)

    # when / then
    with pytest.raises(IndexError):
        coco_dataset[index]


def test_load_image(coco_dataset: service.CoCoTrainingDataset) -> None:
    # given
    image_id = 285
    expected_result = Image.open("./data/validation/val2017/000000000285.jpg", "r").convert("RGB")

    # when
    result = coco_dataset._load_image(image_id)

    # then
    assert result == expected_result


def test_load_captions(coco_dataset: service.CoCoTrainingDataset) -> None:
    # given
    image_id = 285
    expected_result = [
        "A big burly grizzly bear is show with grass in the background.",
        "The large brown bear has a black nose.",
        "Closeup of a brown bear sitting in a grassy area.",
        "A large bear that is sitting on grass. ",
        "A close up picture of a brown bear's face.",
    ]

    # when
    result = coco_dataset._load_captions(image_id)

    # then
    assert result == expected_result


def test_load_segmentations(coco_dataset: service.CoCoTrainingDataset) -> None:
    # given
    image_id = 285

    # when
    result = coco_dataset._load_segmentations(image_id)

    # then
    assert list(result.keys()) == ["bear"]
    assert len(result["bear"]) == 1


def test_getitem(coco_dataset: service.CoCoTrainingDataset) -> None:
    # given
    grizzly_index = 1

    expected_image = Image.open("./data/validation/val2017/000000000285.jpg", "r").convert("RGB")
    expected_captions = [
        "A big burly grizzly bear is show with grass in the background.",
        "The large brown bear has a black nose.",
        "Closeup of a brown bear sitting in a grassy area.",
        "A large bear that is sitting on grass. ",
        "A close up picture of a brown bear's face.",
    ]

    # when
    result_image, result_captions, result_segmentation = coco_dataset[grizzly_index]

    # then
    assert result_image == expected_image
    assert result_captions == expected_captions
    assert list(result_segmentation.keys()) == ["bear"]
    assert len(result_segmentation["bear"]) == 1
