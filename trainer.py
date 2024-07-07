from datetime import datetime
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
        config,
        selected_animal_type,
        selected_model_type,
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
        self.config = config

        dataloader_helper = DataLoaderHelper(self.config)
        self.model_visualizer = ModelVisualizer(self.model, dataloader_helper)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.scheduler = ExponentialLR(optimizer, gamma=0.9) if optimizer else None
        self.early_stopper = EarlyStopper(
            patience=self.config["early_stopping_patience"],
            min_delta=self.config["early_stopping_min_delta"],
        )
        self.writer = SummaryWriter()
        self.save_dir = save_dir
        self.global_step = 0
        self.selected_animal_type = selected_animal_type
        self.selected_model_type = selected_model_type

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def train(self):
        """
        Train the model using the specified dataloaders.
        """
        since = time.time()
        best_acc = 0.0
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        experiment_name = f"{current_time}"
        model_dir = os.path.join(self.save_dir, self.selected_model_type)
        animal_dir = os.path.join(model_dir, self.selected_animal_type)
        if not os.path.exists(animal_dir):
            os.makedirs(animal_dir)

        best_model_params_path = os.path.join(animal_dir, f"best_{experiment_name}.pt")
        should_stop = False
        
        # Save the initial model parameters
        torch.save(self.model.state_dict(), best_model_params_path)

        for epoch in range(self.config["epochs_t_v"]):
            if should_stop:
                break
            print(f"\nEpoch {epoch}/{50 - 1}")
            print("---" * 10)

            self.run_epoch(epoch, "train") # Fase di training
            val_loss, val_acc = self.run_epoch(epoch, "val") #Fase di validation

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), best_model_params_path)

            if self.early_stopper.early_stop(val_loss):
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

    def run_epoch(self, epoch, phase):
        if phase == "train":
            print(f"INIZIO FASE {phase}")
            self.model.train()
        else:
            print(f"INIZIO FASE {phase}")
            self.model.eval()

        running_loss = 0.0
        running_corrects = 0

        for batch_number, (inputs, labels) in enumerate(self.dataloaders[phase]):
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

            self.log_function(f"{phase.format()} Batch {batch_number} Loss: {loss}")

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
        
        print(f"EPOCH {epoch} --> Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n")
        self.writer.add_scalar(f"{phase}/Loss", epoch_loss, epoch)
        self.writer.add_scalar(f"{phase}/Accuracy", epoch_acc, epoch)

        if phase == "train":
            self.scheduler.step()

        return epoch_loss, epoch_acc
