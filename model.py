import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    """Model made by Andy. Going deep with Convolutional, but smaller."""

    def __init__(self, input_shape, num_classes: int = 1000, dropout: float = 0.25):
        """Contstructor."""
        super().__init__()


        self.features = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(input_shape[0], 64, 3, padding=1),
                nn.MaxPool2d(2, 2),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.MaxPool2d(2, 2),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.MaxPool2d(2, 2),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),

                nn.Conv2d(512, 512, 3, padding=1),
                nn.MaxPool2d(2, 2),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),

                nn.Conv2d(1024, 1024, 4, padding=1),
                nn.MaxPool2d(2, 2),
                nn.ReLU(inplace=True),
            ),
        )

        feature_output_size = self._get_flatten_size(input_shape)

        self.classifier = nn.Sequential(
            nn.Linear(feature_output_size, 5120),
            nn.ReLU(),

            nn.Linear(5120, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout / 4),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=1)
        )


    def _get_flatten_size(self, input_shape):
        x = torch.zeros((1, *input_shape))
        with torch.no_grad():
            output = self.features(x)
        batch_size, channels, height, width = output.size()
        return channels * height * width

    def forward(self, x):
        """Call the forward method to back propagate through the model."""
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x



######################################################################################
#                                     TESTS
######################################################################################
# import pytest


# @pytest.fixture(scope="session")
# def data_loaders():
#     from .data import get_data_loaders

#     return get_data_loaders(batch_size=2)


# def test_model_construction(data_loaders):

#     model = MyModel(num_classes=23, dropout=0.3)

#     dataiter = iter(data_loaders["train"])
#     images, labels = dataiter.next()

#     out = model(images)

#     assert isinstance(
#         out, torch.Tensor
#     ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

#     assert out.shape == torch.Size(
#         [2, 23]
#     ), f"Expected an output tensor of size (2, 23), got {out.shape}"
