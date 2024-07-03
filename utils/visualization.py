import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from DataLoaderHelper import DataLoaderHelper

# Setup device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelVisualizer:
    def __init__(self, model, data_loader_helper: DataLoaderHelper):
        self.model = model
        self.data_loader_helper = data_loader_helper
        self.device = device
        self.model.to(self.device)
        self.all_classes = data_loader_helper.index_to_class

    def images_to_probs(self, images):
        """
        Generates predictions and corresponding probabilities from a trained
        network and a list of images.
        """
        images = images.to(self.device)
        output = self.model(images)
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.cpu().numpy())
        return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

    def plot_classes_preds(self, images, labels, selected_classes=None):
        """
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        """

        classes = selected_classes if selected_classes else self.all_classes

        preds, probs = self.images_to_probs(images)

        fig = plt.figure(figsize=(10, 10))
        for idx in range(4):
            ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
            self.data_loader_helper.matplotlib_imshow(images[idx], one_channel=False)
            ax.set_title(
                "{0}, {1:.1f}%\n(label: {2})".format(
                    self.all_classes[preds[idx]],
                    probs[idx] * 100.0,
                    classes[labels[idx].item()],
                ),
                color=("green" if preds[idx] == labels[idx].item() else "red"),
            )
        return fig
