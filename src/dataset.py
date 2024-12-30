import os
from typing import Dict, List, Optional, Tuple
from datasets import Dataset, load_dataset
from PIL import Image
import torch

class LatexOCRDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.instruction = "Write the LaTeX representation for this image."

    def load_datasets(self) -> Tuple[Dataset, Dataset]:
        """Load train, validation and test datasets"""
        dataset_dict = load_dataset(self.cfg.dataset.name)
        
        train_dataset = self._process_split(dataset_dict[self.cfg.dataset.train_split])
        val_dataset = self._process_split(dataset_dict[self.cfg.dataset.validation_split])
        # test_dataset = self._process_split(dataset_dict[self.cfg.dataset.test_split])
        
        return train_dataset, val_dataset

    def _process_split(self, dataset: Dataset) -> Dataset:
        """Process a dataset split"""
        processed_dataset = dataset.map(
            self._convert_sample,
            remove_columns=dataset.column_names,
            num_proc=self.cfg.dataset.num_proc
        )
        return processed_dataset

    def _convert_sample(self, sample: Dict) -> Dict:
        """Convert a single sample to the required conversation format"""
        image = sample[self.cfg.dataset.image_field]
        if isinstance(image, str):  # If image is a path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, dict):  # If image is already loaded
            image = Image.fromarray(image['bytes']).convert('RGB')
        
        conversation = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.instruction},
                        {"type": "image", "image": image}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": sample[self.cfg.dataset.latex_field]}
                    ]
                }
            ]
        }
        return conversation

    @staticmethod
    def compute_metrics(eval_preds) -> Dict[str, float]:
        """Compute evaluation metrics"""
        predictions, labels = eval_preds
        # For now, just return the loss which is computed automatically
        # Add custom metrics here if needed
        return {}