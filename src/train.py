import hydra
from omegaconf import DictConfig
import argparse
from unsloth import unsloth_train
from dataset import LatexOCRDataset
from model import LatexOCRModel
from trainer import LatexOCRTrainer
from utils import Utils
import torch
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Latex OCR model')
    parser.add_argument('--config_path', type=str, default="config",
                       help='Path to Hydra config directory')
    parser.add_argument('--config_name', type=str, default="config",
                       help='Name of the config file')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='HuggingFace token for pushing models')
    return parser.parse_args()

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    args = parse_args()
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    
    logger.info("Setting up training...")
    Utils.set_seed(cfg.training.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Initialize dataset
        logger.info("Loading datasets...")
        dataset_module = LatexOCRDataset(cfg)
        train_dataset, val_dataset = dataset_module.load_datasets()

        # Initialize model and tokenizer
        logger.info("Initializing model...")
        model_module = LatexOCRModel(cfg)
        model, tokenizer = model_module.setup()
        model = model_module.prepare_for_training(model)
        model = model.to(device)

        # Initialize trainer
        logger.info("Setting up trainer...")
        trainer = LatexOCRTrainer(cfg, model, tokenizer, train_dataset, val_dataset)

        # Start training
        logger.info("Starting training...")
        training_stats = trainer.train()
        
        logger.info("Training completed successfully!")
        logger.info(f"Training stats: {training_stats}")


    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()