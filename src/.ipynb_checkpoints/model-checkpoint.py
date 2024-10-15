import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()

        # Define the feature extraction layers
        self.features = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces spatial dimensions by half

            # Second Convolutional Block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces spatial dimensions by half

            # Third Convolutional Block
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces spatial dimensions by half

            # Fourth Convolutional Block
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # Reduces spatial dimensions by half
        )

        # Adaptive pooling to reduce feature map size to (1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Define the classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),  # Dropout to prevent overfitting
            nn.Linear(512, 256),    # Reduce to a smaller dimensional space
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)  # Final output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.features(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)

        # Flatten the tensor before the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.classifier(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
