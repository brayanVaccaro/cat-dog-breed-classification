import threading
import torch
from tkinter import messagebox, filedialog
from DataLoaderHelper import DataLoaderHelper
from model_factory import ModelFactory
from tester import Tester
from trainer import Trainer


class TrainingManager:
    def __init__(self, config, update_log, data_loader, model_factory: ModelFactory):
        self.config = config
        self.update_log = update_log
        self.data_loader = data_loader
        self.model_factory = model_factory
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.training_thread = None
        self.testing_thread = None
        self.trained_model = None
        self.selected_model_type = None
        self.selected_classes = []

    def train_model(self):
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
        trainer = Trainer(
            model,
            train_loader,
            train_loader_sizes,
            optimizer,
            self.selected_classes,
            self.update_log,
            self.config,
        )
        self.trained_model = trainer.train()

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

    def start_training_thread(self, selected_classes, selected_model_type):
        self.selected_classes = selected_classes
        self.selected_model_type = selected_model_type
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
