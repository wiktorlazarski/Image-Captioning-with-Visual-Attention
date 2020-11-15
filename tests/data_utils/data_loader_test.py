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
    expected_result = Image.open("./data/validation/val2017/000000000285.jpg", 'r').convert("RGB")

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
    expected_result = {
        'bear': [[37.31, 373.02, 57.4, 216.61, 67.44, 159.21, 77.49, 113.29, 91.84, 86.03, 123.41,
                84.59, 162.15, 96.07, 215.25, 86.03, 261.17, 70.24, 285.56, 68.81, 337.22, 68.81,
                411.84, 93.2, 454.89, 107.55, 496.5, 255.35, 513.72, 262.53, 552.47, 292.66, 586.0,
                324.23, 586.0, 381.63, 586.0, 449.08, 586.0, 453.38, 578.3, 616.97, 518.03, 621.27,
                444.84, 624.14, 340.09, 625.58, 136.32, 625.58, 1.43, 632.75, 7.17, 555.26, 5.74,
                414.64]]
        }

    # when
    result = coco_dataset._load_segmentations(image_id)

    # then
    assert result == expected_result


def test_getitem(coco_dataset: service.CoCoTrainingDataset) -> None:
    # given
    grizzly_index = 1
    expected_result = (
        Image.open("./data/validation/val2017/000000000285.jpg", 'r').convert("RGB"),
        [
            "A big burly grizzly bear is show with grass in the background.",
            "The large brown bear has a black nose.",
            "Closeup of a brown bear sitting in a grassy area.",
            "A large bear that is sitting on grass. ",
            "A close up picture of a brown bear's face.",
        ],
        {'bear': [[37.31, 373.02, 57.4, 216.61, 67.44, 159.21, 77.49, 113.29, 91.84, 86.03, 123.41,
                84.59, 162.15, 96.07, 215.25, 86.03, 261.17, 70.24, 285.56, 68.81, 337.22, 68.81,
                411.84, 93.2, 454.89, 107.55, 496.5, 255.35, 513.72, 262.53, 552.47, 292.66, 586.0,
                324.23, 586.0, 381.63, 586.0, 449.08, 586.0, 453.38, 578.3, 616.97, 518.03, 621.27,
                444.84, 624.14, 340.09, 625.58, 136.32, 625.58, 1.43, 632.75, 7.17, 555.26, 5.74,
                414.64]]
        }
    )

    # when
    result = coco_dataset[grizzly_index]

    # then
    assert result == expected_result
