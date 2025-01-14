import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel
from huggingface_hub import HfFolder, login
import os
import logging
from typing import Optional
from eval import LatexOCREvaluator
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class ModelDeployer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def setup_huggingface_auth(self, token: Optional[str] = None):
        """Setup authentication with HuggingFace Hub"""
        if token is None:
            token = os.getenv("HF_TOKEN")
        
        if token is None:
            raise ValueError("HuggingFace token not found in environment variables or parameters")
            
        login(token)
        logger.info("Successfully logged into HuggingFace Hub")

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
        merged_model = PeftModel.from_pretrained(model,self.cfg.model.finetune.adapter_path)
        merged_model = merged_model.merge_and_unload()
        
        logger.info("Successfully attached adapter to model")
        return merged_model

    def push_to_hub(self, model, processor):
        """Push the model and processor to HuggingFace Hub"""
        repo_id = self.cfg.huggingface.repo_id
        
        logger.info(f"Pushing model and processor to {repo_id}")
        
        # Push model
        model.push_to_hub(
            repo_id,
            commit_message="Update model with fine-tuned adapter",
            private=self.cfg.huggingface.private,
        )
        
        # Push processor
        processor.push_to_hub(
            repo_id,
            commit_message="Update processor configuration",
            private=self.cfg.huggingface.private,
        )
        
        logger.info("Successfully pushed model and processor to HuggingFace Hub")

    def deploy(self):
        """Complete deployment pipeline"""
        try:
            # Setup authentication
            self.setup_huggingface_auth()
            
            # Load model and processor
            model, processor = self.load_model_and_processor()
            
            # Attach adapter
            model = self.attach_adapter(model)
            
            # Push to hub
            self.push_to_hub(model, processor)
            
            return model, processor
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            raise


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    
    # Deploy model
    deployer = ModelDeployer(cfg)
    model, processor = deployer.deploy()


if __name__ == "__main__":
    main()