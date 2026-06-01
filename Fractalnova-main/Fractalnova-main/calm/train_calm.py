import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from .config import CALMConfig
from .calm_model import CALMModel


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=2048):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings["input_ids"][idx].clone(),
        }


class CALMTrainer:
    def __init__(
        self,
        model: CALMModel,
        config: CALMConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.accelerator = Accelerator(
            log_with="tensorboard",
            project_dir=config.logging_dir,
            gradient_accumulation_steps=config.grad_accumulation_steps,
        )
        self.writer = SummaryWriter(log_dir=config.logging_dir)

    def train(self):
        config = self.config
        model = self.model

        trainable_params = model.get_trainable_parameters()
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )

        total_steps = config.max_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
        )

        model, optimizer, train_loader, scheduler = self.accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )

        global_step = 0
        total_loss = 0.0
        best_eval_loss = float("inf")

        model.train()
        self.accelerator.print(f"Starting CALM training for {config.max_steps} steps")
        self.accelerator.print(f"Trainable parameters: {model.total_trainable:,}")

        while global_step < config.max_steps:
            for batch in train_loader:
                if global_step >= config.max_steps:
                    break

                with self.accelerator.accumulate(model):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs["loss"]
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(trainable_params, 1.0)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    global_step += 1
                    total_loss += loss.detach().float()

                    if global_step % config.logging_steps == 0:
                        avg_loss = total_loss / config.logging_steps
                        perplexity = math.exp(avg_loss)
                        self.accelerator.print(
                            f"Step {global_step}/{config.max_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"Perplexity: {perplexity:.2f} | "
                            f"LR: {scheduler.get_last_lr()[0]:.2e}"
                        )
                        self.writer.add_scalar("loss/train", avg_loss, global_step)
                        self.writer.add_scalar("perplexity/train", perplexity, global_step)
                        total_loss = 0.0

                    if global_step % config.save_steps == 0:
                        self._save_checkpoint(global_step)

                    if (
                        config.eval_steps > 0
                        and global_step % config.eval_steps == 0
                        and self.eval_dataset is not None
                    ):
                        eval_loss = self._evaluate()
                        self.accelerator.print(f"Eval loss: {eval_loss:.4f}")
                        self.writer.add_scalar("loss/eval", eval_loss, global_step)

                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            self._save_checkpoint(global_step, is_best=True)

        self._save_checkpoint(global_step, is_final=True)
        self.accelerator.print("CALM training complete!")
        self.writer.close()

    def _evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
        )

        for batch in eval_loader:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                total_loss += outputs["loss"].item()
                num_batches += 1

        self.model.train()
        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, step: int, is_best: bool = False, is_final: bool = False):
        output_dir = self.config.output_dir
        if is_final:
            subdir = os.path.join(output_dir, "final")
        elif is_best:
            subdir = os.path.join(output_dir, "best")
        else:
            subdir = os.path.join(output_dir, f"checkpoint-{step}")

        os.makedirs(subdir, exist_ok=True)

        bridge_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                bridge_state[name] = param.detach().cpu()

        torch.save(
            {
                "step": step,
                "bridge_state_dict": bridge_state,
                "config": self.model.config,
            },
            os.path.join(subdir, "calm_bridge.pt"),
        )

        self.model.tokenizer.save_pretrained(subdir)
        self.accelerator.print(f"Checkpoint saved to {subdir}")


def train_calm(
    train_texts: list[str],
    eval_texts: Optional[list[str]] = None,
    config: Optional[CALMConfig] = None,
) -> CALMModel:
    if config is None:
        config = CALMConfig()

    model = CALMModel(config)

    train_dataset = TextDataset(train_texts, model.tokenizer, config.anchor_max_length)
    eval_dataset = None
    if eval_texts:
        eval_dataset = TextDataset(eval_texts, model.tokenizer, config.anchor_max_length)

    trainer = CALMTrainer(model, config, train_dataset, eval_dataset)
    trainer.train()

    return model
