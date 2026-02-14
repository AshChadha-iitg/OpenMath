"""
OpenMath Training Script
Fine-tune Qwen2.5-Math-1.5B on GSM8K dataset using QLoRA (4-bit quantization)
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
import argparse
from typing import Dict


class GSM8KTrainer:
    """Trainer class for fine-tuning on GSM8K dataset with QLoRA"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-1.5B",
        output_dir: str = "./checkpoints",
        num_samples: int = 1000,
        max_length: int = 1024,
        num_epochs: int = 6,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
    ):
        """
        Initialize the trainer with configuration parameters

        Args:
            model_name: Base model identifier from HuggingFace
            output_dir: Directory to save model checkpoints
            num_samples: Number of training samples to use from GSM8K
            max_length: Maximum sequence length for tokenization
            num_epochs: Number of training epochs
            lora_rank: LoRA attention dimension
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout probability for LoRA layers
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size per device
            gradient_accumulation_steps: Steps to accumulate gradients
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None

    def load_model_and_tokenizer(self):
        """Load the base model in 4-bit quantization and prepare for training"""
        print(f"Loading model: {self.model_name}")

        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        print("Model loaded successfully in 4-bit mode")
        
    def configure_lora(self):
        """Configure and apply LoRA adapters to the model"""
        print("Configuring LoRA...")
        
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
    def prepare_dataset(self) -> Dict:
        """
        Load and prepare GSM8K dataset for training
        
        Returns:
            Dictionary containing train and eval datasets
        """
        print(f"Loading GSM8K dataset (using {self.num_samples} samples)...")
        
        # Load GSM8K dataset
        dataset = load_dataset("openai/gsm8k", "main")
        
        # Select subset for training
        train_dataset = dataset["train"].shuffle(seed=42).select(range(self.num_samples))
        
        # Use a small portion for validation
        eval_size = min(100, self.num_samples // 10)
        eval_dataset = dataset["test"].shuffle(seed=42).select(range(eval_size))
        
        # Preprocess datasets
        train_dataset = train_dataset.map(
            self.preprocess_function,
            remove_columns=train_dataset.column_names,
            desc="Preprocessing train dataset"
        )
        
        eval_dataset = eval_dataset.map(
            self.preprocess_function,
            remove_columns=eval_dataset.column_names,
            desc="Preprocessing eval dataset"
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Evaluation samples: {len(eval_dataset)}")
        
        return {
            "train": train_dataset,
            "eval": eval_dataset
        }
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """
        Preprocess GSM8K examples into training format
        
        Format: Question: {question}\nAnswer: {answer}
        
        Args:
            examples: Dictionary containing 'question' and 'answer' fields
            
        Returns:
            Tokenized inputs with labels for training
        """
        # Format the prompt
        prompt = (
            "### Instruction:\n"
            "Solve the math problem step by step and give the final answer.\n\n"
            "### Problem:\n"
            f"{examples['question']}\n\n"
            "### Solution:\n"
            f"{examples['answer']}"
        )
        
        # Tokenize
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = tokenized["input_ids"].copy()

        solution_prefix = "### Solution:\n"
        solution_ids = self.tokenizer(
            solution_prefix,
            add_special_tokens=False
        )["input_ids"]

        start_idx = None
        for i in range(len(labels) - len(solution_ids)):
            if labels[i:i + len(solution_ids)] == solution_ids:
                start_idx = i + len(solution_ids)
                break

        if start_idx is not None:
            labels[:start_idx] = [-100] * start_idx

        tokenized["labels"] = labels
        
        return tokenized
    
    def get_training_arguments(self) -> TrainingArguments:
        """
        Configure training arguments
        
        Returns:
            TrainingArguments object with all training configurations
        """
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            eval_strategy="steps",
            save_total_limit=3,
            fp16=True,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            report_to="none",  # Change to "wandb" if using Weights & Biases
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            push_to_hub=False,
        )
    
    def train(self):
        """Execute the complete training pipeline"""
        # Step 1: Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Step 2: Configure LoRA
        self.configure_lora()
        
        # Step 3: Prepare dataset
        datasets = self.prepare_dataset()
        
        # Step 4: Setup training arguments
        training_args = self.get_training_arguments()
        
        # Step 5: Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["eval"],
        )
        # Step 6: Train
        print("\n" + "="*50)
        print("Starting training...")
        print("="*50 + "\n")
        
        trainer.train()
        
        # Step 7: Save final model
        print("\nSaving LoRA adapter...")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"\n✓ Training complete!")
        print(f"✓ LoRA adapter saved to: {self.output_dir}")
        

def main():
    """Main entry point for training script"""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-Math-1.5B on GSM8K using QLoRA"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Base model name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for model checkpoints"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of training samples to use from GSM8K"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization"
    )
    
    # Training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=6,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA attention dimension"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and run trainer
    trainer = GSM8KTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        max_length=args.max_length,
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
