import time
import torch
import torch.nn as nn
import shutil
from pathlib import Path
from DataLoaderHelper import DataLoaderHelper
from model_factory import ModelFactory
from torch.utils.tensorboard import (
    SummaryWriter,
)  # libreria per loggare info nella tensorboard


class ExperimentRunner:
    """
    Classe 'laboratorio' di esempio.
    Permette di gestire un gruppo di esperimenti eseguiti a diverse condizioni.
    """

    def __init__(self, lab_folder: str, config, selected_classes, loss_fn, update_log, clear_lab: bool = False) -> None:
        """
        Inizializza la cartella esperimenti, eventualmente cancellando i precedenti.

        Args:
            lab_folder (str): Cartella esperimenti.
            clear_lab (bool, optional): Indica se eliminare i precedenti. Defaults to False.
        """
        self.lab_path = Path(lab_folder)

        if clear_lab and self.lab_path.is_dir():
            shutil.rmtree(self.lab_path)

        self.config = config
        self.selected_classes = selected_classes
        # self.model = model
        self.loss_fn = loss_fn
        self.update_log = update_log
        self.data_loader_helper = DataLoaderHelper(self.config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        self.optimizer = None

    def setup_experiment(self, num_epochs: int, l_rate: float, batch_size: int, model) -> None:
        """
        Setta le impostazioni per il prossimo esperimento da eseguire.

        Args:
            model (nn.Module): Modello di rete utilizzato dall'esperimento.
            num_epochs (int): Numero di epoche.
            l_rate (float): Learning rate.
            batch_size (int): Dimensione del batch.
            dataset_size (int): Dimensione del dataset.
        """
        self.model = model
        self.num_epochs = num_epochs
        self.l_rate = l_rate
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=l_rate)
        self.experiment_loader, self.experiment_loader_sizes = self.data_loader_helper.load_experiment_data(
            self.selected_classes, batch_size
        )
        self.update_log(
            f"SIZES:\ntrain -> {self.experiment_loader_sizes['train']}, batch_size={self.experiment_loader['train'].batch_size}\nval -> {self.experiment_loader_sizes['val']}, batch_size={self.experiment_loader['val'].batch_size}\n"
        )
        self.update_log(
            "Experiment started with following params:\n- num_epochs: {}\n- l_rate: {}\n- batch_size = {}".format(num_epochs, l_rate, batch_size)
        )

    def run_experiment(self, exp_name: str):
        """
        Lancia l'esecuzione dell'esperimento con tale descrizione.

        Args:
            exp_name (str): Descrizione esperimento.
            selected_classes (list): Lista delle classi selezionate per il dataset.
        """
        self.exp_path = self.lab_path / exp_name
        self.writer = SummaryWriter(str(self.exp_path))

        # Carica il dataset

        # Definisce la loss function e l'optimizer

        # # Aggiungi il grafo del modello
        # dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        # self.writer.add_graph(self.model, dummy_input)

        monitor = 1
        monitor_print = True  # da capire true o false

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(self.experiment_loader["train"]):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(self.experiment_loader["train"])
            epoch_acc = 100 * correct / total

            self.writer.add_scalar("train/loss", epoch_loss, epoch)
            self.writer.add_scalar("train/accuracy", epoch_acc, epoch)

            if epoch % monitor == monitor - 1 and monitor_print:
                print(
                    f"Epoch: {epoch+1}\t-->\tLoss: {epoch_loss:.8f}\tAccuracy: {epoch_acc:.2f}%"
                )

        # Valutazione sul set di validazione
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in self.experiment_loader["val"]:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(self.experiment_loader["val"])
        val_acc = 100 * val_correct / val_total

        self.writer.add_scalar("val/loss", val_loss, epoch)
        self.writer.add_scalar("val/accuracy", val_acc, epoch)

        # Logga gli iperparametri e i risultati finali
        self.writer.add_hparams(
            {
                "num_epochs": self.num_epochs,
                "learning_rate": self.l_rate,
                "batch_size": self.batch_size,
            },
            {"hparams/best_val_loss": val_loss, "hparams/best_val_acc": val_acc},
        )

        self.writer.flush()
        self.writer.close()



# # Questo codice viene eseguito solo se viene eseguito questo file direttamente
# if __name__ == '__main__':
#     run_experiments()
