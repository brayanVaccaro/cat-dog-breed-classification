from datetime import datetime
import shutil
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
        self.selected_animal_type = selected_animal_type
        self.selected_model_type = selected_model_type

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
        
        self.model_dir = os.path.join(self.save_dir, self.selected_model_type)
        self.animal_dir = os.path.join(self.model_dir, self.selected_animal_type)
        self.checkpoint_path = f"checkpoint.pt"
        self.stop_training_flag = False

        # Add model graph to TensorBoard
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        self.writer.add_graph(self.model, dummy_input)
        self.writer.flush()

    def train(self, start_epoch=0):
        """
        Train the model using the specified dataloaders.
        """
        # Remove the directory if it exists
        if os.path.exists(self.animal_dir):
            shutil.rmtree(self.animal_dir)
        
        # Create the directory
        os.makedirs(self.animal_dir)
        
        since = time.time()
        best_acc = 0.0
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        experiment_name = f"{current_time}"
        
        best_model_params_path = os.path.join(self.animal_dir, f"best_{experiment_name}.pt")
        
        should_stop = False
        
        # Save the initial model parameters
        if start_epoch == 0:
            torch.save(self.model.state_dict(), best_model_params_path)

        for epoch in range(start_epoch, self.config["epochs_t_v"]):
            if should_stop:
                break
            
            if self.stop_training_flag:
                self.save_checkpoint(epoch)
                self.log_function("Training stopped by user")
                break
            
            self.log_function(f"\nEpoch {epoch}/{self.config['epochs_t_v'] - 1}")
            self.log_function("---" * 10)

            self.run_epoch(epoch, "train") # Fase di training
            val_loss, val_acc = self.run_epoch(epoch, "val") #Fase di validation

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), best_model_params_path)

            if self.early_stopper.early_stop(val_loss):
                self.log_function("Early stopping triggered")
                should_stop = True
                break

        time_elapsed = time.time() - since
        self.log_function(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        self.log_function(f"Best val Acc: {best_acc:.4f}")

        # Load the best model parameters saved during training
        self.model.load_state_dict(torch.load(best_model_params_path))

        self.writer.close()
        return self.model

    def run_epoch(self, epoch, phase):
        self.log_function(f"INIZIO FASE {phase}")
        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_corrects = 0
        embeddings = []
        labels_list = []
        inputs_list = []

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

            if phase == "val":
                embeddings.append(outputs)
                labels_list.append(labels)
                inputs_list.append(inputs)

            print(f"{phase.format()} Batch {batch_number} Loss: {loss}")

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
        
        self.log_function(f"EPOCH {epoch} --> Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n")
        self.writer.add_scalar(f"{phase}/Loss", epoch_loss, epoch)
        self.writer.add_scalar(f"{phase}/Accuracy", epoch_acc, epoch)

        if phase == "train":
            self.scheduler.step()
        else:
            # Log embeddings to TensorBoard
            embeddings = torch.cat(embeddings).cpu()
            labels_list = torch.cat(labels_list).cpu()
            inputs_list = torch.cat(inputs_list).cpu()
            
            class_names = self.selected_classes
            labels_name = [class_names[i] for i in labels_list]
            metadata=[f'{label}:{name}' for label,name in zip(labels_list, labels_name)]
            self.writer.add_embedding(
                embeddings,
                metadata=metadata,
                label_img=inputs_list,
                global_step=self.global_step
            )

        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(state, os.path.join(self.animal_dir, self.checkpoint_path))
        self.log_function(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        self.log_function(f"Checkpoint loaded from epoch {epoch}")
        return epoch
    
    def stop(self):
        self.stop_training_flag = True