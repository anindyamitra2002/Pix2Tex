from typing import Dict, Tuple
from datasets import Dataset, load_dataset
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)
class LatexOCRDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.instruction = "Write the LaTeX representation for this image."
        self.train_size = self.cfg.dataset.train_size
        self.val_size = self.cfg.dataset.val_size

    def load_datasets(self):
        """Load train, validation and test datasets"""
        dataset_dict = load_dataset(self.cfg.dataset.name)
        train_dataset = dataset_dict[self.cfg.dataset.train_split]
        if self.train_size and self.train_size < len(train_dataset):
            logger.info(f"Limiting train dataset from {len(train_dataset)} to {self.train_size} samples")
            train_dataset = train_dataset.select(range(self.train_size))
        
        # Load and limit validation dataset
        val_dataset = dataset_dict[self.cfg.dataset.validation_split]
        if self.val_size and self.val_size < len(val_dataset):
            logger.info(f"Limiting validation dataset from {len(val_dataset)} to {self.val_size} samples")
            val_dataset = val_dataset.select(range(self.val_size))
        
        train_dataset = self._process_split(train_dataset)
        val_dataset = self._process_split(val_dataset)
        # test_dataset = self._process_split(dataset_dict[self.cfg.dataset.test_split])
        
        return train_dataset, val_dataset

    def _process_split(self, dataset: Dataset):
        """Process a dataset split"""
        return [self._convert_sample(sample) for sample in dataset]
        

    def _convert_sample(self, sample: Dict):
        """Convert a single sample to the required conversation format"""
        image = sample[self.cfg.dataset.image_field]
        image_path = os.path.join(self.cfg.dataset.image_dir, image)
        
        conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.instruction},
                        {"type": "image", "image": image_path}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": sample[self.cfg.dataset.latex_field]}
                    ]
                }
            ]
        return conversation

    @staticmethod
    def compute_metrics(eval_preds) -> Dict[str, float]:
        """Compute evaluation metrics"""
        predictions, labels = eval_preds
        # For now, just return the loss which is computed automatically
        # Add custom metrics here if needed
        return {}