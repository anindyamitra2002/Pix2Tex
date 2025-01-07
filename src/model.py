from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

class LatexOCRModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.processor = None

    def setup(self):
        """Initialize and setup the model and processor"""
        # Setup BitsAndBytes config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.cfg.model.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Initialize model and processor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.cfg.model.name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2"
            
        )
        
        processor = Qwen2VLProcessor.from_pretrained(self.cfg.model.name, min_pixels=256*28*28, max_pixels=512*28*28, padding_side="right")

        # Apply PEFT
        model, peft_config = self._apply_peft(model)
        
        return model, processor, peft_config

    def _apply_peft(self, model: Qwen2VLForConditionalGeneration):
        """Apply PEFT configurations to the model"""
        peft_config = LoraConfig(
            lora_alpha=self.cfg.model.finetune.lora_alpha,
            lora_dropout=self.cfg.model.finetune.lora_dropout,
            r=self.cfg.model.finetune.r,
            bias=self.cfg.model.finetune.bias,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]  # Specific to Qwen2VL
        )
        
        model = get_peft_model(model, peft_config)
        return model, peft_config