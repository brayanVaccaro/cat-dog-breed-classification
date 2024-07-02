from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
import time
import os
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard.writer import SummaryWriter
from tempfile import TemporaryDirectory

from torch import Tensor
from DataLoaderHelper import DataLoaderHelper
from EarlyStopper import EarlyStopper
from utils.visualization import ModelVisualizer


class Trainer:
    def __init__(
        self,
        model,
        dataloaders,
        dataset_sizes,
        optimizer,
        selected_classes,
        log_function,
        save_dir="./model_weights",
    ):
        """
        Initialize the Trainer class with model, dataloaders, dataset sizes, optimizer, selected classes and logging function.
        """
        self.model = model
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.optimizer = optimizer
        self.selected_classes = selected_classes
        self.log_function = log_function
        dataloader_helper = DataLoaderHelper()
        self.model_visualizer = ModelVisualizer(self.model, dataloader_helper)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.cuda(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.scheduler = ExponentialLR(optimizer, gamma=0.9) if optimizer else None
        self.early_stopper = EarlyStopper(patience=3, min_delta=0.03)
        self.writer = SummaryWriter()
        self.save_dir = save_dir
        self.global_step = 0

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def train(self):
        """
        Train the model using the specified dataloaders.
        """
        since = time.time()
        best_acc = 0.0

        best_model_params_path = os.path.join(self.save_dir, "best_model_params.pt")
        should_stop = False

        # Save the initial model parameters
        torch.save(self.model.state_dict(), best_model_params_path)

        for epoch in range(50):
            if should_stop:
                break
            print(f"\nEpoch {epoch}/{50 - 1}")
            print("---" * 10)

            for phase in ["train", "val"]:
                if phase == "train":
                    print(f"INIZIO FASE {phase}")
                    self.model.train()
                else:
                    print(f"INIZIO FASE {phase}")
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for batch_number, (inputs, labels) in enumerate(
                    self.dataloaders[phase]
                ):
                    self.global_step += 1
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    self.log_function(
                        f"{phase.format()} Batch {batch_number} Loss: {loss}"
                    )

                    if phase == "val" and batch_number % 5 == 0:
                        self.writer.add_figure(
                            "predictions vs. actuals",
                            self.model_visualizer.plot_classes_preds(
                                inputs, labels, self.selected_classes
                            ),
                            global_step=self.global_step,
                        )

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                print(
                    f"EPOCH {epoch} --> Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n"
                )
                self.writer.add_scalar(f"{phase}/Loss", epoch_loss, epoch)
                self.writer.add_scalar(f"{phase}/Accuracy", epoch_acc, epoch)

                if phase == "train":
                    self.scheduler.step()

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(self.model.state_dict(), best_model_params_path)

            if phase == "val":
                if self.early_stopper.early_stop(epoch_loss):
                    print("Early stopping triggered")
                    should_stop = True
                    break

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {best_acc:.4f}")

        # Load the best model parameters saved during training
        self.model.load_state_dict(torch.load(best_model_params_path))

        self.writer.close()
        return self.model

    def test(self, test_loader):
        """
        Test the model using the specified dataloader.
        """
        self.model.eval()
        running_corrects = 0

        all_preds = []
        all_labels = []

        for inputs, labels in test_loader:
            inputs: Tensor = inputs.to(self.device)
            labels: Tensor = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        total = len(test_loader.dataset)
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
