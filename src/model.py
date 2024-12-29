from unsloth import FastVisionModel
from transformers import AutoTokenizer
from typing import Tuple

class LatexOCRModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None

    def setup(self) -> Tuple[FastVisionModel, AutoTokenizer]:
        """Initialize and setup the model and tokenizer"""
        model, tokenizer = FastVisionModel.from_pretrained(
            self.cfg.model.name,
            load_in_4bit=self.cfg.model.load_in_4bit,
            use_gradient_checkpointing=self.cfg.model.use_gradient_checkpointing
        )

        model = self._apply_peft(model)
        return model, tokenizer

    def _apply_peft(self, model: FastVisionModel) -> FastVisionModel:
        """Apply PEFT configurations to the model"""
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=self.cfg.model.finetune.vision_layers,
            finetune_language_layers=self.cfg.model.finetune.language_layers,
            finetune_attention_modules=self.cfg.model.finetune.attention_modules,
            finetune_mlp_modules=self.cfg.model.finetune.mlp_modules,
            r=self.cfg.model.finetune.r,
            lora_alpha=self.cfg.model.finetune.lora_alpha,
            lora_dropout=self.cfg.model.finetune.lora_dropout,
            bias=self.cfg.model.finetune.bias,
            random_state=self.cfg.model.finetune.random_state,
            use_rslora=self.cfg.model.finetune.use_rslora,
            loftq_config=self.cfg.model.finetune.loftq_config
        )
        return model

    @staticmethod
    def prepare_for_training(model: FastVisionModel) -> FastVisionModel:
        """Prepare model for training"""
        return FastVisionModel.for_training(model)

    @staticmethod
    def prepare_for_inference(model: FastVisionModel) -> FastVisionModel:
        """Prepare model for inference"""
        return FastVisionModel.for_inference(model)