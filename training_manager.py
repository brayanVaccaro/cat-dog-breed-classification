import os
import threading
import torch
from tkinter import messagebox, filedialog
from DataLoaderHelper import DataLoaderHelper
from model_factory import ModelFactory
from tester import Tester
from trainer import Trainer
from experiment import ExperimentRunner  # Importa la funzione che esegue gli esperimenti

class TrainingManager:
    def __init__(self, config, update_log, data_loader: DataLoaderHelper, model_factory: ModelFactory):
        self.config = config
        self.update_log = update_log
        self.data_loader = data_loader
        self.model_factory = model_factory
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.training_thread = None
        self.testing_thread = None
        self.experiment_thread = None  # Thread per gli esperimenti
        self.trained_model = None
        self.selected_model_type = None
        self.selected_classes = []

    def train_model(self, resume=False):
      # Preparazione per il training
        self.update_log(
            "Training started with selected classes... ({})".format(
                self.selected_classes
            )
        )

        train_loader, train_loader_sizes = self.data_loader.load_data(
            self.selected_classes
        )

        self.update_log(
            f"SIZES:\ntrain -> {train_loader_sizes['train']}, batch_size={train_loader['train'].batch_size}\nval -> {train_loader_sizes['val']}, batch_size={train_loader['val'].batch_size}\n"
        )

        n_classes = len(self.selected_classes)

        # Utilizzo di ModelFactory per creare il modello

        model = self.load_model()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["optimizer_lr"])

        # Creazione di un'istanza di Trainer e inizio del training
        self.trainer = Trainer(
            model,
            train_loader,
            train_loader_sizes,
            optimizer,
            self.selected_classes,
            self.update_log,
            self.config,
            self.selected_animal_type,
            self.selected_model_type,
        )
        
        if resume:
            checkpoint_path = os.path.join(self.trainer.animal_dir, self.trainer.checkpoint_path)
            start_epoch = self.trainer.load_checkpoint(checkpoint_path)
        else:
            start_epoch = 0
        
        self.trained_model = self.trainer.train(start_epoch=start_epoch)

        self.update_log("Training completed.")

    def test_model(self):
        """Test the trained model."""
        if not self.trained_model:
            if messagebox.askyesno(
                "Model not found",
                "No trained model found. Do you want to load a saved model?",
            ):
                model_path = filedialog.askopenfilename(
                    filetypes=[("PyTorch models", "*.pt")]
                )
                if model_path:
                    self.load_model(model_path)
                else:
                    self.update_log("Model loading cancelled.")
                    return
            else:
                self.update_log("Please train the model first.")
                return

        self.update_log(
            "Testing started with selected classes... ({})".format(
                self.selected_classes
            )
        )
        test_loader, test_loader_size = self.data_loader.load_test_data(
            self.selected_classes
        )

        tester = Tester(
            self.trained_model,
            test_loader,
            self.selected_classes,
            self.device,
            self.update_log,
        )

        accuracy, all_preds, all_labels = tester.test()
        self.update_log(f"Test Accuracy: {accuracy:.4f}")

    def load_model(self, model_path=""):
        n_classes = len(self.selected_classes)
        
        if self.selected_model_type == "ResNet50":
            model = self.model_factory.get_resnet50(n_classes).to(self.device)
        elif self.selected_model_type == "AlexNet":
            model = self.model_factory.get_alexnet(n_classes).to(self.device)
        
        if self.phase == 'Test':
            model.load_state_dict(torch.load(model_path))
            self.trained_model = model
            self.update_log(f"Model loaded from {model_path}")
            return
        
        return model

    def start_training_thread(self, selected_classes, selected_model_type, selected_animal_type):
        self.selected_classes = selected_classes
        self.selected_model_type = selected_model_type
        self.selected_animal_type = selected_animal_type
        self.phase = "Train"
        if not self.selected_classes:
            messagebox.showerror("Error", "No classes selected")
            return
        print("Selected classes:", self.selected_classes)

        self.training_thread = threading.Thread(target=self.train_model, daemon=True)
        self.training_thread.start()

    def start_testing_thread(self, selected_classes, selected_model_type):
        self.selected_classes = selected_classes
        self.selected_model_type = selected_model_type
        self.phase = "Test"

        if not self.selected_classes:
            messagebox.showerror("Error", "No classes selected")
            return
        print("Selected classes:", self.selected_classes)
        
        self.testing_thread = threading.Thread(target=self.test_model, daemon=True)
        self.testing_thread.start()


    def run_experiment(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        
        experiment = ExperimentRunner('./runs', self.config, self.selected_classes, loss_fn, self.update_log, True)
        
        # Lista del numero di epoche da testare
        exp_epochs = [100]  # Accorciato per test rapidi

        # Lista dei learning rate da testare
        exp_lrs = [0.001, 0.0001]

        # Lista delle dimensioni dei batch e delle dimensioni del dataset
        exp_batch_sizes = [32, 64]
        exp_dataset_sizes = [100, 300, 500, 1000, 1500]

        # Cicla attraverso le configurazioni e esegui gli esperimenti
        exp_counter = 0
        for epochs in exp_epochs:
            for lr in exp_lrs:
                for batch_size in exp_batch_sizes:
                    model = self.load_model()
                    experiment.setup_experiment(
                        num_epochs=epochs, l_rate=lr, batch_size=batch_size, model=model
                    )
                    experiment.run_experiment(f"test_{exp_counter}")
                    exp_counter += 1
                    self.update_log(f" - Esperimento {exp_counter}: completato.")

        print("Lab: chiuso")

        self.update_log("Esperimenti terminati.")

    def start_experiment_thread(self, selected_classes, selected_model_type,selected_animal_type):
        self.selected_classes = selected_classes
        self.selected_model_type = selected_model_type
        self.selected_animal_type = selected_animal_type

        self.phase = "Experiment"
        if self.experiment_thread and self.experiment_thread.is_alive():
            self.update_log("Experiment is already in progress.")
            return

        self.experiment_thread = threading.Thread(target=self.run_experiment)
        self.experiment_thread.start()

    def resume_training_thread(self, selected_classes, selected_model_type, selected_animal_type):
        self.selected_classes = selected_classes
        self.selected_model_type = selected_model_type
        self.selected_animal_type = selected_animal_type
        self.phase = "Train"
        if not self.selected_classes:
            messagebox.showerror("Error", "No classes selected")
            return
        print("Resuming training with classes:", self.selected_classes)

        self.training_thread = threading.Thread(target=lambda: self.train_model(resume=True), daemon=True)
        self.training_thread.start()

    def stop_training(self):
        """Stop the training thread."""
        if self.training_thread.is_alive():
            self.trainer.stop()