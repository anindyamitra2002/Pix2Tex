import torch
from tqdm import tqdm
from jiwer import wer, cer
from typing import List, Dict
from sacrebleu.metrics import BLEU
import logging
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from dataset import LatexOCRDataset
logger = logging.getLogger(__name__)

class LatexOCREvaluator:
    def __init__(self, model, processor, cfg, device="cuda"):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.device = device
        self.batch_size = cfg.eval.batch_size
        self.max_new_tokens = cfg.eval.max_new_tokens
        self.bleu = BLEU()

    def load_test_data(self):
        """Load and process test dataset"""
        dataset_dict = load_dataset(self.cfg.dataset.name)
        test_dataset = dataset_dict[self.cfg.dataset.test_split]
        dataset = LatexOCRDataset(self.cfg)
        return dataset._process_split(test_dataset)

    def batch_inference(self, batch_samples: List) -> List[str]:
        """Perform batch inference on a list of samples"""
        # Extract conversations from samples
        conversations = [sample[0] for sample in batch_samples]  # Skip system message
        print(conversations)
        # Prepare text inputs
        texts = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
        
        # Process vision information
        image_inputs, _ = process_vision_info(batch_samples)
        
        # Prepare model inputs
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate predictions
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens
            )
        
        # Trim and decode outputs
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return outputs

    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute evaluation metrics"""
        metrics = {}
        
        # BLEU score
        bleu_score = self.bleu.corpus_score([pred.split() for pred in predictions],
                                          [[ref.split()] for ref in references])
        metrics['bleu'] = bleu_score.score
        
        # Character Error Rate
        metrics['cer'] = cer(references, predictions)
        
        # Word Error Rate
        metrics['wer'] = wer(references, predictions)
        
        return metrics

    def evaluate(self):
        """Run evaluation on test dataset"""
        # Load test data
        logger.info("Loading test dataset...")
        test_samples = self.load_test_data()
        
        # Initialize predictions and references lists
        all_predictions = []
        all_references = []
        
        # Create batches
        num_samples = len(test_samples)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Starting evaluation on {num_samples} samples...")
        
        # Process batches with progress bar
        with tqdm(total=num_samples, desc="Evaluating") as pbar:
            for i in range(0, num_samples, self.batch_size):
                batch = test_samples[i:i + self.batch_size]
                
                # Get predictions for batch
                predictions = self.batch_inference(batch)
                all_predictions.extend(predictions)
                
                # Extract references from batch
                references = [
                    sample[1]["content"][0]["text"] 
                    for sample in batch
                ]
                all_references.extend(references)
                
                # Update progress bar
                pbar.update(len(batch))
                
                # Log intermediate metrics every N batches
                if (i // self.batch_size + 1) % self.cfg.eval.log_every_n_batches == 0:
                    metrics = self.compute_metrics(all_predictions[-len(batch):], 
                                                all_references[-len(batch):])
                    logger.info(f"Batch {i // self.batch_size + 1}/{num_batches} metrics: {metrics}")
        
        # Compute final metrics
        final_metrics = self.compute_metrics(all_predictions, all_references)
        
        # Log final results
        logger.info("Evaluation completed!")
        logger.info(f"Final metrics: {final_metrics}")
        
        return final_metrics, all_predictions, all_references

