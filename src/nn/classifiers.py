import torch
import torchvision


class ResNet18(torch.nn.Module):
    """
    Modification to ResNet encoder is adapted from
    https://github.com/leftthomas/SimCLR/blob/master/model.py
    """

    def __init__(self, num_classes: int = 10) -> None:
        super(ResNet18, self).__init__()
        self.num_classes = num_classes

        # Isolate the ResNet model into an encoder and a linear classifier.

        # Get the correct dimensions of the classifer.
        self.encoder = torchvision.models.resnet18(
            num_classes=self.num_classes)
        self.linear_in_features = self.encoder.fc.in_features
        self.linear_out_features = self.encoder.fc.out_features
        self.encoder.fc = torch.nn.Identity()

        # Modify the encoder.
        del self.encoder
        self.encoder = []
        for name, module in torchvision.models.resnet18(
                num_classes=self.num_classes).named_children():
            if name == 'conv1':
                module = torch.nn.Conv2d(3,
                                         64,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         bias=False)
            if not isinstance(module, torch.nn.Linear) and not isinstance(
                    module, torch.nn.MaxPool2d):
                self.encoder.append(module)
        self.encoder.append(torch.nn.Flatten())
        self.encoder = torch.nn.Sequential(*self.encoder)

        # This is the linear classifier for fine-tuning and inference.
        self.linear = torch.nn.Linear(in_features=self.linear_in_features,
                                      out_features=self.linear_out_features)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.linear(self.encoder(x))

    def init_linear(self):
        torch.nn.init.constant_(self.linear.weight, 0.01)
        torch.nn.init.constant_(self.linear.bias, 0)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(
                    m, torch.nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                    torch.nn.init.constant_(m.bias, 0)


class SmallConvNet(torch.nn.Module):

    def __init__(self,
                 num_classes: int = 10,
                 image_shape: str = (3, 32, 32)) -> None:
        super(SmallConvNet, self).__init__()
        self.num_classes = num_classes

        # Get the correct dimensions of the classifer.
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 256, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(256, 256, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten()
        )

        sample_input = torch.ones((1, *image_shape))
        sample_output = self.encoder(sample_input)
        assert len(sample_output.shape) == 2

        # This is the linear classifier for fine-tuning and inference.
        self.linear = torch.nn.Linear(in_features=sample_output.shape[-1],
                                      out_features=self.num_classes)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.linear(self.encoder(x))

    def init_linear(self):
        torch.nn.init.constant_(self.linear.weight, 0.01)
        torch.nn.init.constant_(self.linear.bias, 0)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(
                    m, torch.nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                    torch.nn.init.constant_(m.bias, 0)
