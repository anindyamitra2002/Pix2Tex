from unsloth.trainer import UnslothVisionDataCollator
from unsloth import is_bf16_supported
from trl import SFTTrainer, SFTConfig
from unsloth import unsloth_train
from typing import Dict, Any
from huggingface_hub import HfFolder
import os

class LatexOCRTrainer:
    def __init__(self, cfg, model, tokenizer, train_dataset, eval_dataset):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        # self._init_wandb()
        self._setup_hf_token()
        
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            entity=self.cfg.wandb.entity,
            config=self.cfg
        )

    def _setup_hf_token(self):
        """Setup HuggingFace token for model pushing"""
        if "HF_TOKEN" in os.environ:
            HfFolder.save_token(os.environ["HF_TOKEN"])
        elif self.cfg.huggingface.token:
            HfFolder.save_token(self.cfg.huggingface.token)

    def setup_trainer(self) -> SFTTrainer:
        """Setup the SFT trainer"""
        training_args = SFTConfig(
            per_device_train_batch_size=self.cfg.training.batch_size,
            per_device_eval_batch_size=self.cfg.training.batch_size,
            gradient_accumulation_steps=self.cfg.training.gradient_accumulation_steps,
            warmup_steps=self.cfg.training.warmup_steps,
            num_train_epochs=self.cfg.training.num_train_epochs,
            learning_rate=self.cfg.training.learning_rate,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=self.cfg.training.weight_decay,
            lr_scheduler_type=self.cfg.training.lr_scheduler_type,
            seed=self.cfg.training.seed,
            output_dir=self.cfg.paths.output_dir,
            report_to="wandb",
            
            # Evaluation settings
            evaluation_strategy=self.cfg.training.evaluation_strategy,
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
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            args=training_args,
        )
        
        return trainer

    def train(self) -> Dict[str, Any]:
        """Run the training process"""
        trainer = self.setup_trainer()
        training_stats = unsloth_train(trainer)
        
        # Push the best model to HuggingFace Hub
        trainer.push_to_hub()
        
        # wandb.finish()
        return training_stats

    @staticmethod
    def compute_metrics(eval_preds):
        """Compute evaluation metrics"""
        predictions, labels = eval_preds
        # Return empty dict as loss is computed automatically
        return {}