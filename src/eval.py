import torch
from tqdm import tqdm
from typing import List
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from dataset import LatexOCRDataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel
import hydra
from omegaconf import DictConfig
import os
import logging

logger = logging.getLogger(__name__)
class ModelLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    def load_model_and_processor(self):
        """Load the base model and processor"""
        logger.info(f"Loading base model from {self.cfg.model.name}")
        
        # Load model with mixed precision
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.cfg.model.name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        # Load processor
        processor = Qwen2VLProcessor.from_pretrained(self.cfg.model.name)
        
        logger.info("Successfully loaded base model and processor")
        return model, processor

    def attach_adapter(self, model):
        """Attach the trained adapter to the model"""
        logger.info(f"Loading adapter from {self.cfg.huggingface.repo_id}")
        merged_model = PeftModel.from_pretrained(model,"/teamspace/studios/this_studio/Pix2Tex/checkpoints/checkpoint-500/")
        merged_model = merged_model.merge_and_unload()
        
        logger.info("Successfully attached adapter to model")
        return merged_model
    
class LatexOCREvaluator:
    def __init__(self, model, processor, cfg, device="cuda"):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.device = device
        self.batch_size = cfg.eval.batch_size
        self.max_new_tokens = cfg.eval.max_new_tokens
        self.test_size = self.cfg.eval.test_size

    def load_test_data(self):
        """Load and process test dataset"""
        dataset_dict = load_dataset(self.cfg.dataset.name)
        test_dataset = dataset_dict[self.cfg.dataset.test_split]
        if self.test_size and self.test_size < len(test_dataset):
            logger.info(f"Limiting test dataset from {len(test_dataset)} to {self.test_size} samples")
            test_dataset = test_dataset.select(range(self.test_size))
        dataset = LatexOCRDataset(self.cfg)
        return dataset._process_split(test_dataset)

    def batch_inference(self, batch_samples: List) -> List[str]:
        """Perform batch inference on a list of samples"""
        try:
            # Extract conversations from samples
            conversations = [[sample[0]] for sample in batch_samples]

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
            
        except Exception as e:
            logger.error(f"Error in batch inference: {str(e)}")
            return [""] * len(batch_samples)  # Return empty strings for failed batch

    def evaluate(self):
        """Run evaluation and save predictions"""
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
        with tqdm(total=num_samples, desc="Generating predictions") as pbar:
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
                    
                except Exception as e:
                    logger.error(f"Error processing batch {i // self.batch_size + 1}: {str(e)}")
                    continue
        
        # Save predictions and references
        self.save_results(all_predictions, all_references)
        
        logger.info("Evaluation completed!")
        return all_predictions, all_references

    def save_results(self, predictions: List[str], references: List[str]):
        """Save predictions and references to files"""
        try:
            output_dir = self.cfg.eval.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Save predictions
            with open(os.path.join(output_dir, self.cfg.eval.pred_file), 'w') as f:
                for pred in predictions:
                    f.write(f"{pred}\n")
            
            # Save references
            with open(os.path.join(output_dir, self.cfg.eval.ref_file), 'w') as f:
                for ref in references:
                    f.write(f"{ref}\n")
                    
            logger.info(f"Results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

# Example usage
def evaluate_model(model, processor, cfg):
    evaluator = LatexOCREvaluator(model, processor, cfg)
    predictions, references = evaluator.evaluate()
    return predictions, references

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    loader = ModelLoader(cfg)
    model, processor = loader.load_model_and_processor()
    # Attach adapter
    model = loader.attach_adapter(model)
    predictions, references = evaluate_model(model, processor, cfg)