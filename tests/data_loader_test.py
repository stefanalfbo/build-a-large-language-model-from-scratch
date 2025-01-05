import torch

from src.data_loader import create_dataloader_v1


def test_create_dataloader_v1():
    # Arrange
    expected_input_token_ids = torch.tensor([[40, 367, 2885, 1464]])
    expected_target_token_ids = torch.tensor([[367, 2885, 1464, 1807]])
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    # Act
    data_iter = iter(dataloader)
    first_batch = next(data_iter)

    # Assert
    assert torch.equal(first_batch[0], expected_input_token_ids)
    assert torch.equal(first_batch[1], expected_target_token_ids)


def test_create_dataloader_with_max_length_2_stride_2():
    # Arrange
    expected_input_token_ids = torch.tensor(
        [[40, 367], [2885, 1464], [1807, 3619], [402, 271]]
    )
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    dataloader = create_dataloader_v1(
        raw_text, batch_size=4, max_length=2, stride=2, shuffle=False
    )

    # Act
    data_iter = iter(dataloader)
    first_batch = next(data_iter)

    # Assert
    assert torch.equal(first_batch[0], expected_input_token_ids)


def test_create_dataloader_with_max_length_8_stride_2():
    # Arrange
    expected_input_token_ids = torch.tensor(
        [
            [40, 367, 2885, 1464, 1807, 3619, 402, 271],
            [2885, 1464, 1807, 3619, 402, 271, 10899, 2138],
            [1807, 3619, 402, 271, 10899, 2138, 257, 7026],
            [402, 271, 10899, 2138, 257, 7026, 15632, 438],
        ]
    )

    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    dataloader = create_dataloader_v1(
        raw_text, batch_size=4, max_length=8, stride=2, shuffle=False
    )

    # Act
    data_iter = iter(dataloader)
    first_batch = next(data_iter)

    # Assert
    assert torch.equal(first_batch[0], expected_input_token_ids)
