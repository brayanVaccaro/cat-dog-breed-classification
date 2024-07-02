import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.alexnet import AlexNet_Weights

class ModelFactory:
    def __init__(self, device):
        """
        Initialize the ModelFactory with a specific device.
        
        Args:
        - device (torch.device): The device to deploy the model to.
        """
        self.device = device

    def get_resnet50(self, num_classes):
        """
        Return a ResNet50 model adapted for a specific number of classes.
        """
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, 37)
        return model.to(self.device)

    def get_alexnet(self, num_classes):
        """
        Return an AlexNet model adapted for a specific number of classes.
        """
        model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 37)
        return model.to(self.device)
