from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import Tensor
import numpy as np
from matplotlib import pyplot as plt, ticker

from DataLoaderHelper import DataLoaderHelper
from utils.visualization import ModelVisualizer
from torch.utils.tensorboard.writer import SummaryWriter
class Tester:
    def __init__(
        self,
        model,
        dataloader,
        selected_classes,
        device,
        log_function,
    ):
        self.model = model
        self.dataloader = dataloader
        self.selected_classes = selected_classes
        self.device = device
        self.log_function = log_function
        self.writer = SummaryWriter()
        dataloader_helper = DataLoaderHelper()
        self.model_visualizer = ModelVisualizer(self.model, dataloader_helper)
        self.global_step = 0

    def test(self):
        self.model.eval()
        running_corrects = 0
        all_preds = []
        all_labels = []

        for inputs, labels in self.dataloader:
            self.global_step += 1
            inputs: Tensor = inputs.to(self.device)
            labels: Tensor = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if self.global_step % 5 == 0:
                self.writer.add_figure(
                    "TEST predictions vs. actuals",
                    self.model_visualizer.plot_classes_preds(
                        inputs, labels, self.selected_classes
                    ),
                    global_step=self.global_step,
                )

        total = len(self.dataloader.dataset)
        accuracy = running_corrects.double() / total
        print(f"Test Accuracy: {accuracy:.4f}")

        # Analyze results and log them
        self.analyze_results(all_preds, all_labels)
        return accuracy, all_preds, all_labels

    def analyze_results(self, all_preds, all_labels):
        """
        Analyze the results of the predictions and log them.
        """
        report = classification_report(
            all_labels, all_preds, target_names=self.selected_classes, output_dict=True
        )
        matrix = confusion_matrix(all_labels, all_preds)

        precision = []
        recall = []
        f1_score = []
        # Extract metrics for each class
        for name in self.selected_classes:
            precision.append(report[name]["precision"])
            recall.append(report[name]["recall"])
            f1_score.append(report[name]["f1-score"])

        # Log precision, recall, and f1-score for each class
        self.plot_metrics(precision, "Precision")
        self.plot_metrics(recall, "Recall")
        self.plot_metrics(f1_score, "F1 Score")

        # Log confusion matrix as image
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(matrix, cmap=plt.cm.get_cmap("gist_rainbow"))
        fig.colorbar(cax)

        plt.xlabel("Predicted")
        plt.ylabel("True")
        self.writer.add_figure("Confusion Matrix", fig, global_step=self.global_step)

        print(
            "Classification Report:\n",
            classification_report(
                all_labels, all_preds, target_names=self.selected_classes
            ),
        )
        print("Confusion Matrix:\n", matrix)

    def plot_metrics(self, values, metric_name):
        """
        Plot the given metrics for all classes.
        """
        fig, ax = plt.subplots(figsize=(10, 20))

        # Creazione delle barre orizzontali con valori e indici corretti
        indices = np.arange(len(self.selected_classes))
        bar_width = 0.6  # Imposta la larghezza delle barre
        ax.barh(indices, values, align="center", height=bar_width)

        # Impostazione delle etichette sugli assi
        ax.set_yticks(indices)
        ax.set_yticklabels(self.selected_classes)

        # Aggiunta delle etichette e del titolo
        ax.set_xlabel(metric_name)
        ax.set_title(f"{metric_name} for each class")

        # Imposta gli step sull'asse X ogni 0.1
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

        # Aggiunta di linee a ogni step sull'asse X
        ax.grid(axis="x", which="both", linestyle="--", linewidth=0.5)

        # Log della figura su TensorBoard
        self.writer.add_figure(
            f"{metric_name} per class", fig, global_step=self.global_step
        )

        # Chiusura della figura per evitare che matplotlib ne accumuli in memoria
        plt.close(fig)
