from trl import SFTTrainer, SFTConfig
from typing import Dict, Any
from huggingface_hub import HfFolder
from qwen_vl_utils import process_vision_info
import os
import wandb

class LatexOCRTrainer:
    def __init__(self, cfg, model, processor, train_dataset, eval_dataset, peft_config):
        self.cfg = cfg
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.peft_config = peft_config
        self._init_wandb()
        self._setup_hf_token()

    def _setup_hf_token(self):
        """Setup HuggingFace token for model pushing"""
        if "HF_TOKEN" in os.environ:
            HfFolder.save_token(os.environ["HF_TOKEN"])
        elif self.cfg.huggingface.token:
            HfFolder.save_token(self.cfg.huggingface.token)
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name
        )


    def collate_fn(self, examples):
        """Custom collate function for processing batches"""
        # Get the texts and images, and apply the chat template
        texts = [
            self.processor.apply_chat_template(example, tokenize=False)
            for example in examples
        ]
        # Process images using process_vision_info
        image_inputs = [
            process_vision_info([example])[0]
            for example in examples
        ]

        # Tokenize the texts and process the images
        batch = self.processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True
        )
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Ignore the image token indices in the loss computation (Qwen2VL specific)
        image_tokens = [151652, 151653, 151655]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels
        return batch

    def setup_trainer(self) -> SFTTrainer:
        """Setup the SFT trainer"""
        training_args = SFTConfig(
            per_device_train_batch_size=self.cfg.training.batch_size,
            per_device_eval_batch_size=self.cfg.training.batch_size,
            gradient_accumulation_steps=self.cfg.training.gradient_accumulation_steps,
            warmup_steps=self.cfg.training.warmup_steps,
            max_steps=self.cfg.training.num_train_steps,
            learning_rate=self.cfg.training.learning_rate,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=self.cfg.training.weight_decay,
            lr_scheduler_type=self.cfg.training.lr_scheduler_type,
            seed=self.cfg.training.seed,
            output_dir=self.cfg.paths.output_dir,
            report_to="wandb",
            
            # Evaluation settings
            eval_strategy=self.cfg.training.evaluation_strategy,
            eval_steps=self.cfg.training.eval_steps,
            
            # Saving settings
            save_strategy=self.cfg.training.save_strategy,
            save_steps=self.cfg.training.save_steps,
            save_total_limit=self.cfg.training.save_total_limit,
            load_best_model_at_end=self.cfg.training.load_best_model_at_end,
            metric_for_best_model=self.cfg.training.metric_for_best_model,
            greater_is_better=self.cfg.training.greater_is_better,
            
            # HuggingFace Hub settings
            push_to_hub=True,
            hub_model_id=self.cfg.huggingface.repo_id,
            hub_private_repo=self.cfg.huggingface.private,
            
            # Vision finetuning specific settings
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=self.cfg.dataset.num_proc,
            max_seq_length=self.cfg.training.max_seq_length,
        )

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.processor.tokenizer,
            data_collator=self.collate_fn,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            peft_config=self.peft_config,
            args=training_args
        )
        
        return trainer

    def train(self) -> Dict[str, Any]:
        """Run the training process"""
        
        trainer = self.setup_trainer()
        train_result = trainer.train()
        
        # Push the best model to HuggingFace Hub
        trainer.push_to_hub()
        wandb.finish()
        return train_result.metrics