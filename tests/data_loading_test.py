import pytest
import scripts.data_loading as dl
import torch


@pytest.fixture
def coco_dataset() -> dl.CocoCaptions:
    dset_type = dl.DatasetType.VALIDATION
    return dl.CocoCaptions(dset_paths=dl.TRAINING_DATASET_PATHS[dset_type])


def test_coco_dataset_len(coco_dataset: dl.CocoCaptions) -> None:
    # given
    expected_result = 25_014

    # when
    result = len(coco_dataset)

    # then
    assert result == expected_result


def test_coco_dataset_sorted_by_captions(coco_dataset: dl.CocoCaptions) -> None:
    # given
    examine_first = 500

    # when
    lengths = []
    for i in range(examine_first):
        _, caption = coco_dataset[i]
        lengths.append(len(caption.split()))

    # then
    assert lengths == sorted(lengths, reverse=True)
