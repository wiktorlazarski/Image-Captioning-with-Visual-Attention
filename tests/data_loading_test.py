import scripts.data_loading as dl
import torch


def test_load_dataset() -> None:
    # given
    expected_size = 5_000
    expected_out_tensor_dim = torch.Size([3, 224, 224])

    # when
    dataset = dl.load_dataset(service.DatasetType.Validation, vgg_preprocessed=True)

    # then
    assert len(dataset) == expected_size
    assert dataset[0][0].shape == expected_out_tensor_dim
