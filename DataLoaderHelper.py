import os
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
import torch.utils.data
from torchvision.transforms import v2

from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class DataLoaderHelper:
    def __init__(self, config, root_dir="./data"):
        self.root_dir = root_dir
        self.config = config
        self.annotations_file = os.path.join(
            root_dir, "oxford-iiit-pet", "annotations", "trainval.txt"
        )
        self.data_transforms = {
            "train": v2.Compose(
                [
                    v2.RandomResizedCrop(size=(224, 224)),
                    v2.RandomHorizontalFlip(),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": v2.Compose(
                [
                    v2.Resize(size=(224, 224)),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
            "test": v2.Compose(
                [
                    v2.Resize(size=(224, 224)),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                ]
            ),
        }

        self.dataset = self.create_oxford_train()
        self.classes_to_idx = self.dataset.class_to_idx

    def create_oxford_train(self):
        return OxfordIIITPet(
            root=self.root_dir, transform=self.data_transforms["train"], download=True
        )

    def load_data(self, selected_classes=None):
        self.selected_classes = selected_classes
        return self.filter_and_split_dataset()

    def filter_and_split_dataset(self):
        filtered_class_to_idx, selected_indices, updated_labels = self.filter_dataset(
            self.dataset
        )

        train_indices, val_indices, train_labels, val_labels = self.split_dataset(
            selected_indices, updated_labels
        )

        self.print_class_distribution(train_indices, train_labels, "Train")
        self.print_class_distribution(val_indices, val_labels, "Validation")

        train_dataset = FilteredDataset(
            self.dataset, train_indices, train_labels, self.data_transforms["train"]
        )
        val_dataset = FilteredDataset(
            self.dataset, val_indices, val_labels, self.data_transforms["val"]
        )

        dataloaders = {
            "train": DataLoader(
                train_dataset,
                batch_size=self.config["batch_size_t"],
                shuffle=True,
                num_workers=4,
            ),
            "val": DataLoader(
                val_dataset,
                batch_size=self.config["batch_size_v"],
                shuffle=False,
                num_workers=4,
            ),
        }
        dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

        return dataloaders, dataset_sizes

    def filter_dataset(self, dataset: OxfordIIITPet):
        filtered_class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.selected_classes)
        }

        class_indices = {class_name: [] for class_name in self.selected_classes}

        for i, (_, label) in enumerate(dataset):
            class_name = dataset.classes[label]
            if class_name in filtered_class_to_idx:
                class_indices[class_name].append(i)

        selected_indices = [
            idx for indices in class_indices.values() for idx in indices
        ]
        updated_labels = [
            filtered_class_to_idx[dataset.classes[dataset[idx][1]]]
            for idx in selected_indices
        ]

        assert all(
            0 <= label < len(self.selected_classes) for label in updated_labels
        ), "Some labels are out of the expected range!"

        return filtered_class_to_idx, selected_indices, updated_labels

    def split_dataset(self, selected_indices, updated_labels):

        train_indices, val_indices = train_test_split(
            selected_indices, test_size=0.2, stratify=updated_labels, random_state=42
        )

        train_labels = [
            updated_labels[selected_indices.index(i)] for i in train_indices
        ]
        val_labels = [updated_labels[selected_indices.index(i)] for i in val_indices]

        return train_indices, val_indices, train_labels, val_labels

    def print_class_distribution(self, indices, labels, dataset_type="Train"):
        from collections import Counter

        label_counts = Counter(labels)
        print(f"{dataset_type} dataset distribution:")
        for class_idx, count in label_counts.items():
            class_name = [
                name
                for idx, name in enumerate(self.selected_classes)
                if idx == class_idx
            ]
            print(f"Class {class_name} (index {class_idx}): {count} images")

    def load_test_data(self, selected_classes):
        self.selected_classes = selected_classes
        dataset = OxfordIIITPet(
            root=self.root_dir,
            split="test",
            transform=self.data_transforms["test"],
            download=True,
        )

        filtered_class_to_idx, selected_indices, updated_labels = self.filter_dataset(
            dataset
        )

        test_labels = [
            updated_labels[selected_indices.index(i)] for i in selected_indices
        ]

        self.print_class_distribution(selected_indices, test_labels, "Test")

        test_dataset = FilteredDataset(
            dataset, selected_indices, test_labels, self.data_transforms["test"]
        )
        dataloader = DataLoader(
            test_dataset,
            batch_size=self.config["batch_size_t"],
            shuffle=True,
            num_workers=4,
        )
        # self.display_all_images(dataloader) #DEBUG
        return dataloader, len(test_dataset)

    def matplotlib_imshow(self, img: Tensor):

        # Denormalizza l'immagine
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (
            img.cpu() * std[:, None, None] + mean[:, None, None]
        )  # Applica la denormalizzazione su ciascun canale
        npimg = img.numpy()
        # Clipping dei valori per essere nell'intervallo corretto
        npimg = np.clip(npimg, 0, 1)
        # Trasposizione per convertire l'immagine da (C, H, W) a (H, W,
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def breeds(self):
        cat_data = []
        dog_data = []
        with open(self.annotations_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[1]) - 1  # CLASS-ID - 1
                species = int(parts[2])
                if species == 1:  # Gatto
                    cat_data.append(class_id)
                elif species == 2:  # Cane
                    dog_data.append(class_id)

        # Creazione delle liste di razze di gatti e cani con indici corrispondenti
        cat_breeds = {
            breed: idx for breed, idx in self.classes_to_idx.items() if idx in cat_data
        }
        dog_breeds = {
            breed: idx for breed, idx in self.classes_to_idx.items() if idx in dog_data
        }

        return dog_breeds, cat_breeds


# Creazione di un nuovo dataset filtrato con etichette aggiornate
class FilteredDataset(Dataset):
    def __init__(
        self, original_dataset, selected_indices, updated_labels, transform=None
    ):
        self.original_dataset = original_dataset
        self.selected_indices = selected_indices
        self.updated_labels = updated_labels
        self.transform = transform

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        # Estrae i dati dall'indice originale e sostituisce l'etichetta con quella aggiornata
        original_idx = self.selected_indices[idx]
        data, _ = self.original_dataset[original_idx]
        label = self.updated_labels[idx]
        if self.transform:
            data = self.transform(data)
        return data, label
