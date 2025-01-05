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


def test_token_id_to_embedding_vector_conversion():
    input_ids = torch.tensor([2, 3, 5, 1])
    expected = torch.tensor([[-0.4015, 0.9666, -1.1481]])
    vocabulary_size = 6
    output_dimensions = 3

    # instantiate the embedding layer
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocabulary_size, output_dimensions)
    # assert embedding_layer.weight == torch.tensor([
    #     [ 0.3374, -0.1778, -0.1690],
    #     [ 0.9178,  1.5810,  1.3010],
    #     [ 1.2753, -0.2010, -0.1606],
    #     [-0.4015,  0.9666, -1.1481], <--- this is the one we want to test
    #     [-1.1589,  0.3255, -0.6315],
    #     [-2.8400, -0.7849, -1.4096]
    # ], requires_grad=True))
    result = embedding_layer(torch.tensor([3]))

    # Assert
    assert result.shape == expected.shape
    # torch.tensor([[-0.4015,  0.9666, -1.1481]])

    # apply the embedding layer to the input_ids
    result = embedding_layer(input_ids)

    # Assert
    # tensor([
    #     [ 1.2753, -0.2010, -0.1606],
    #     [-0.4015,  0.9666, -1.1481],
    #     [-2.8400, -0.7849, -1.4096],
    #     [ 0.9178,  1.5810,  1.3010]
    # ]
    assert result.shape == (4, 3)
