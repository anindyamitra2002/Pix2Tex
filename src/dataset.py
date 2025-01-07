from typing import Dict, Tuple
from datasets import Dataset, load_dataset
from PIL import Image
import os
class LatexOCRDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.instruction = "Write the LaTeX representation for this image."

    def load_datasets(self):
        """Load train, validation and test datasets"""
        dataset_dict = load_dataset(self.cfg.dataset.name)
        
        train_dataset = self._process_split(dataset_dict[self.cfg.dataset.train_split])
        val_dataset = self._process_split(dataset_dict[self.cfg.dataset.validation_split])
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