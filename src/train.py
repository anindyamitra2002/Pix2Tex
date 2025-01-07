import hydra
from omegaconf import DictConfig
from dataset import LatexOCRDataset
from model import LatexOCRModel
from trainer import LatexOCRTrainer
from logger import Utils
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    
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

        # Initialize model and processor
        logger.info("Initializing model...")
        model_module = LatexOCRModel(cfg)
        model, processor, peft_config = model_module.setup()
        model = model.to(device)

        # Initialize trainer
        logger.info("Setting up trainer...")
        trainer = LatexOCRTrainer(cfg, model, processor, train_dataset, val_dataset, peft_config)

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