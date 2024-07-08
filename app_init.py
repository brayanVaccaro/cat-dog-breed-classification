import json
import tkinter as tk
from tkinter import Frame, messagebox, scrolledtext
import threading
from tkinter import filedialog
import torch
from DataLoaderHelper import DataLoaderHelper
from model_factory import ModelFactory
from trainer import Trainer
from sklearn.metrics import classification_report, confusion_matrix
from training_manager import TrainingManager

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Class Selector for OxfordIIITPet Dataset")
        # Parametri dal file config.json
        with open("config.json", "r") as f:
            self.config = json.load(f)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_loader = DataLoaderHelper(self.config, root_dir="./data")
        self.model_factory = ModelFactory(device=self.device)

        self.dog_classes, self.cat_classes = self.data_loader.breeds()
        self.selected_animal_type = None
        self.selected_classes = []
        self.available_models = {0: "ResNet50", 1: "AlexNet"}
        # Caricamento delle classi del dataset
        self.classes = self.data_loader.classes_to_idx
        self.training_manager = TrainingManager(
            self.config, self.update_log, self.data_loader, self.model_factory
        )

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
        # Abilita lo scorrimento con la rotellina del mouse
        self.class_canvas.bind(
            "<MouseWheel>",
            lambda event: self.class_canvas.yview_scroll(
                int(-1 * (event.delta / 120)), "units"
            ),
        )

        self.class_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        tk.Label(
            self.scrollable_class_frame, text="Dog Breeds", font=("Helvetica", 18)
        ).pack()
        # Checkbox per selezionare tutte le classi
        self.select_all_var = tk.BooleanVar()
        self.select_all_checkbox = tk.Checkbutton(
            self.scrollable_class_frame,
            text="Select All",
            variable=self.select_all_var,
            command=self.toggle_all_classes,
            font=("Helvetica", 16),
        )
        self.select_all_checkbox.pack(anchor="w")
        # Creazione dei checkbox per la selezione delle classi
        self.checkboxes = []
        self.class_vars = []

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

    def toggle_all_classes(self):
        """Toggle the state of all class checkboxes based on the 'Select All' checkbox."""
        for var in self.class_vars:
            var.set(self.select_all_var.get())

    def setup_buttons(self):
        """Setup the start and stop training buttons."""
        self.button_frame = Frame(self.root)
        self.button_frame.pack(side="bottom", fill="both", expand=False)

        self.train_button = tk.Button(
            self.button_frame,
            text="Start Training",
            command=lambda: self.open_model_choice_dialog("Train"),
        )
        self.train_button.pack(side="left", padx=10, pady=10)
        
        # self.stop_button = tk.Button(
        #     self.button_frame, text="Stop Training", command=self.stop_training
        # )
        # self.stop_button.pack(side="right", padx=10, pady=10)
        # self.stop_button.config(state=tk.DISABLED)

        self.test_button = tk.Button(
            self.button_frame,
            text="Test Model",
            command=lambda: self.open_model_choice_dialog("Test"),
        )
        self.test_button.pack(side="left", padx=10, pady=10)
        
        # Bottone per eseguire gli esperimenti
        self.experiment_button = tk.Button(
            self.button_frame,
            text="Run Experiments",
            command=lambda: self.open_model_choice_dialog("Experiment"),
        )
        self.experiment_button.pack(side="left", padx=10, pady=10)

    def start_training_thread(self):
        """Start the training in a separate thread."""
        self.selected_classes = [
            cls for var, cls in zip(self.class_vars, self.classes) if var.get()
        ]
        self.train_button.config(state=tk.DISABLED)
        # self.stop_button.config(state=tk.NORMAL)
        self.training_manager.start_training_thread(
            self.selected_classes, self.selected_model_type, self.selected_animal_type
        )
        self.train_button.config(state=tk.NORMAL)
        self.test_button.config(state=tk.NORMAL)
        # self.stop_button.config(state=tk.DISABLED)

    def start_testing_thread(self):
        self.selected_classes = [
            cls for var, cls in zip(self.class_vars, self.classes) if var.get()
        ]
        self.test_button.config(state=tk.DISABLED)
        self.train_button.config(state=tk.DISABLED)
        # self.stop_button.config(state=tk.DISABLED)
        self.training_manager.start_testing_thread(
            self.selected_classes, self.selected_model_type
        )
        
    def start_experiment_thread(self):
        """Run experiments in a separate thread."""
        self.selected_classes = [
            cls for var, cls in zip(self.class_vars, self.classes) if var.get()
        ]
        self.training_manager.start_experiment_thread(
           self.selected_classes, self.selected_model_type
           )

    def update_log(self, message):
        """Update the log area in the UI with a new message."""
        self.log_area.config(state="normal")
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.config(state="disabled")

    def open_model_choice_dialog(self, phase):
        """Open a dialog for model selection."""
        self.model_dialog = tk.Toplevel(self.root)
        self.model_dialog.title("Select Model Type")
        self.model_dialog.geometry("300x200")  # Dimensione della finestra
        self.phase = phase
        
        # Impedisce interazioni con la finestra principale
        self.model_dialog.grab_set() 
        tk.Label(
            self.model_dialog, text="Select Model Type", font=("Helvetica", 14)
        ).pack(pady=20)

        self.model_var = tk.IntVar()

        for idx, model_name in self.available_models.items():
            tk.Radiobutton(
                self.model_dialog,
                text=model_name,
                variable=self.model_var,
                value=idx,
                font=("Helvetica", 12),
            ).pack(anchor="w")

        tk.Button(
            self.model_dialog,
            text="Select",
            command=lambda: self.set_model_type(phase),
            font=("Helvetica", 12),
        ).pack(pady=10)

    def set_model_type(self, phase):
        """Set the selected model type and start the corresponding thread."""
        self.selected_model_type = self.available_models[self.model_var.get()]
        self.model_dialog.destroy()
        
        if phase == "Train":
            self.start_training_thread()
        elif phase == "Test":
            self.start_testing_thread()
        elif phase == "Experiment":
            self.start_experiment_thread()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
