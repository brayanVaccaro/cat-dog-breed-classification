import tkinter as tk
from tkinter import Frame, messagebox, scrolledtext
import threading
from tkinter import filedialog
import torch
from DataLoaderHelper import DataLoaderHelper
from model_factory import ModelFactory
from trainer import Trainer
from sklearn.metrics import classification_report, confusion_matrix


class App:
    def __init__(self, root):
        self.root = root
        root.title("Class Selector for OxfordIIITPet Dataset")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_loader = DataLoaderHelper(root_dir="./data")
        self.model_factory = ModelFactory(device=self.device)

        self.dog_classes, self.cat_classes = self.data_loader.breeds()
        self.selected_animal_type = None
        self.training_thread = None
        self.testing_thread = None
        self.trained_model = None
        self.selected_classes = []
        self.index_to_class = {}

        # Caricamento delle classi del dataset
        self.classes = self.data_loader.classes_to_idx

        # Costruzione dell'interfaccia utente con Tkinter
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface for the application."""
        # Finestra di scelta iniziale
        self.setup_initial_choice()

    def setup_log_area(self):
        # Area di log per visualizzare i messaggi di training
        self.log_area = scrolledtext.ScrolledText(
            self.root, state="disabled", height=10, width=120
        )
        self.log_area.pack(side="top", fill="both", expand=True)

    def setup_initial_choice(self):
        """Setup the initial choice window for selecting dogs or cats."""
        self.choice_frame = Frame(self.root)
        self.choice_frame.pack(side="top", fill="both", expand=True)

        tk.Label(
            self.choice_frame, text="Choose Training Type", font=("Helvetica", 18)
        ).pack(pady=20)

        self.dog_button = tk.Button(
            self.choice_frame,
            text="Dogs",
            command=self.choose_dogs,
            font=("Helvetica", 16),
        )
        self.dog_button.pack(side="left", padx=20, pady=20)

        self.cat_button = tk.Button(
            self.choice_frame,
            text="Cats",
            command=self.choose_cats,
            font=("Helvetica", 16),
        )
        self.cat_button.pack(side="right", padx=20, pady=20)

    def choose_dogs(self):
        self.selected_animal_type = "dogs"
        self.classes = self.dog_classes
        self.setup_class_selector()
        self.setup_log_area()
        self.choice_frame.pack_forget()

    def choose_cats(self):
        self.selected_animal_type = "cats"
        self.classes = self.cat_classes
        self.setup_class_selector()
        self.setup_log_area()
        self.choice_frame.pack_forget()

    def setup_class_selector(self):
        """Setup the class selector interface."""
        self.class_frame = Frame(self.root)
        self.class_frame.pack(side="top", fill="both", expand=False)

        self.class_canvas = tk.Canvas(self.class_frame)
        self.scrollbar = tk.Scrollbar(
            self.class_frame, orient="vertical", command=self.class_canvas.yview
        )
        self.scrollable_class_frame = Frame(self.class_canvas)
        self.scrollable_class_frame.bind(
            "<Configure>",
            lambda e: self.class_canvas.configure(
                scrollregion=self.class_canvas.bbox("all")
            ),
        )
        self.class_canvas.create_window(
            (0, 0), window=self.scrollable_class_frame, anchor="nw"
        )
        self.class_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.class_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Creazione dei checkbox per la selezione delle classi
        self.checkboxes = []
        self.class_vars = []
        tk.Label(
            self.scrollable_class_frame, text="Dog Breeds", font=("Helvetica", 18)
        ).pack()
        for cls in self.classes:
            var = tk.BooleanVar()
            cb = tk.Checkbutton(
                self.scrollable_class_frame,
                text=cls,
                variable=var,
                font=("Helvetica", 16),
                justify="left",
            )
            cb.pack(anchor="w")
            self.checkboxes.append(cb)
            self.class_vars.append(var)

        # Pulsanti per iniziare e fermare il training
        self.setup_buttons()

    def setup_buttons(self):
        """Setup the start and stop training buttons."""
        self.button_frame = Frame(self.root)
        self.button_frame.pack(side="bottom", fill="both", expand=False)

        self.train_button = tk.Button(
            self.button_frame, text="Start Training", command=self.start_training_thread
        )
        self.train_button.pack(side="left", padx=10, pady=10)

        # self.stop_button = tk.Button(
        #     self.button_frame, text="Stop Training", command=self.stop_training
        # )
        # self.stop_button.pack(side="right", padx=10, pady=10)
        # self.stop_button.config(state=tk.DISABLED)

        self.test_button = tk.Button(
            self.button_frame, text="Test Model", command=self.start_testing_thread
        )
        self.test_button.config(
            state=tk.NORMAL
        )  # Disabilitato fino al completamento del training
        self.test_button.pack()

    def start_training_thread(self):
        """Start the training in a separate thread."""
        self.selected_classes = [
            cls for var, cls in zip(self.class_vars, self.classes) if var.get()
        ]
        self.index_to_class = {self.classes[cls]: cls for cls in self.selected_classes}
        if not self.selected_classes:
            self.selected_classes = [
                cls for var, cls in zip(self.class_vars, self.classes)
            ]
            # return
        print("Selected indices:", self.index_to_class)
        self.train_button.config(state=tk.DISABLED)
        # self.stop_button.config(state=tk.NORMAL)
        self.training_thread = threading.Thread(target=self.train_model, daemon=True)
        self.training_thread.start()

    def train_model(self):
        # Preparazione per il training
        self.update_log(
            "Training started with selected classes... ({})".format(
                self.index_to_class.values()
            )
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        (train_loader, train_loader_sizes) = self.data_loader.load_data(
            self.selected_classes
        )

        n_classes = len(self.index_to_class) if len(self.index_to_class) > 0 else 12

        # Utilizzo di ModelFactory per creare il modello

        model = self.model_factory.get_resnet50(n_classes)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Creazione di un'istanza di Trainer e inizio del training
        trainer = Trainer(
            model,
            train_loader,
            train_loader_sizes,
            optimizer,
            self.index_to_class,
            self.update_log,
        )
        self.trained_model = trainer.train()

        self.update_log("Training completed.")
        self.train_button.config(state=tk.NORMAL)
        self.test_button.config(state=tk.NORMAL)
        # self.stop_button.config(state=tk.DISABLED)

    def start_testing_thread(self):
        """Start the testing in a separate thread."""
        self.test_button.config(state=tk.DISABLED)
        self.train_button.config(state=tk.DISABLED)
        # self.stop_button.config(state=tk.DISABLED)
        self.testing_thread = threading.Thread(target=self.test_model, daemon=True)

        self.testing_thread.start()

    def test_model(self):
        """Test the trained model."""
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
                self.test_button.config(state=tk.NORMAL)
                return
        else:
            self.update_log("Please train the model first.")
            self.test_button.config(state=tk.NORMAL)
            return
        if not self.selected_classes:
            self.selected_classes = [
                cls for var, cls in zip(self.class_vars, self.classes)
            ]

        self.update_log(
            "Testing started with selected classes... ({})".format(
                self.index_to_class.values()
            )
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_loader, train_loader_size = self.data_loader.load_test_data(
            self.selected_classes
        )

        model = self.trained_model

        trainer = Trainer(
            model, None, None, None, self.selected_classes, self.update_log
        )

        accuracy, all_preds, all_labels = trainer.test(train_loader)

        self.update_log(f"Test Accuracy: {accuracy:.4f}")
        self.test_button.config(state=tk.NORMAL)

    def load_model(self, model_path):
        device = self.device

        n_classes = len(self.index_to_class) if len(self.index_to_class) > 0 else 37
        model = self.model_factory.get_resnet50(n_classes)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        self.trained_model = model
        self.update_log(f"Model loaded from {model_path}")

    def update_log(self, message):
        """Update the log area in the UI with a new message."""
        self.log_area.config(state="normal")
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
