import torch
from tqdm import tqdm
from jiwer import wer, cer
from typing import List, Dict
from sacrebleu.metrics import BLEU
import logging
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from dataset import LatexOCRDataset
import os

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
        self.test_size = self.cfg.eval.test_size

    def load_test_data(self):
        """Load and process test dataset"""
        dataset_dict = load_dataset(self.cfg.dataset.name)
        test_dataset = dataset_dict[self.cfg.dataset.test_split]
        if self.test_size and self.test_size < len(test_dataset):
            logger.info(f"Limiting train dataset from {len(test_dataset)} to {self.test_size} samples")
            test_dataset = test_dataset.select(range(self.test_size))
        dataset = LatexOCRDataset(self.cfg)
        return dataset._process_split(test_dataset)

    def batch_inference(self, batch_samples: List) -> List[str]:
        """Perform batch inference on a list of samples"""
        # Extract conversations from samples
        conversations = [[sample[0]] for sample in batch_samples]  # Skip system message

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
        try:
            metrics = {}
            
            # Ensure predictions and references are not empty
            if not predictions or not references:
                logger.warning("Empty predictions or references received")
                return {
                    'bleu': 0.0,
                    'cer': 1.0,
                    'wer': 1.0
                }
            
            # Clean and prepare predictions and references
            cleaned_preds = [pred.strip() if pred else "" for pred in predictions]
            cleaned_refs = [ref.strip() if ref else "" for ref in references]
            
            # Format references properly for BLEU score
            # BLEU expects: List[List[str]] for references where inner list contains multiple references
            formatted_refs = [[ref] for ref in cleaned_refs]  # Each reference in its own list
            
            # BLEU score - handle empty strings
            valid_pairs = [(p, [r]) for p, r in zip(cleaned_preds, formatted_refs) 
                         if p and r[0]]  # Filter out empty strings
            
            if valid_pairs:
                valid_preds, valid_refs = zip(*valid_pairs)
                bleu_score = self.bleu.corpus_score(valid_preds, valid_refs)
                metrics['bleu'] = bleu_score.score
            else:
                metrics['bleu'] = 0.0
            
            # Character Error Rate
            metrics['cer'] = cer(cleaned_refs, cleaned_preds)
            
            # Word Error Rate
            metrics['wer'] = wer(cleaned_refs, cleaned_preds)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            return {
                'bleu': 0.0,
                'cer': 1.0,
                'wer': 1.0
            }

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
                try:
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
                        batch_metrics = self.compute_metrics(
                            all_predictions[-len(batch):],
                            references
                        )
                        logger.info(f"Batch {i // self.batch_size + 1}/{num_batches} metrics: {batch_metrics}")
                
                except Exception as e:
                    logger.error(f"Error processing batch {i // self.batch_size + 1}: {str(e)}")
                    continue
        
        # Compute final metrics
        final_metrics = self.compute_metrics(all_predictions, all_references)
        
        # Log final results
        logger.info("Evaluation completed!")
        logger.info(f"Final metrics: {final_metrics}")
        
        # Save predictions and references for analysis
        self.save_results(all_predictions, all_references, final_metrics)
        
        return final_metrics, all_predictions, all_references

    def save_results(self, predictions: List[str], references: List[str], metrics: Dict[str, float]):
        """Save evaluation results to files"""
        try:
            output_dir = self.cfg.eval.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Save predictions
            with open(os.path.join(output_dir, 'predictions.txt'), 'w') as f:
                for pred in predictions:
                    f.write(f"{pred}\n")
            
            # Save references
            with open(os.path.join(output_dir, 'references.txt'), 'w') as f:
                for ref in references:
                    f.write(f"{ref}\n")
            
            # Save metrics
            with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value}\n")
                    
            logger.info(f"Results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

