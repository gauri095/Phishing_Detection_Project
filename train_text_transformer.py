"""
Fine-tune DistilBERT (or other HF model) on a CSV with columns: message,label

Usage example:
python train_text_transformer.py --train-csv data/messages_train.csv --valid-csv data/messages_valid.csv --output-dir model/text_distilbert --model-name distilbert-base-uncased --epochs 3
"""
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate
import numpy as np
import os

def preprocess_function(examples, tokenizer, text_col="message"):
    return tokenizer(examples[text_col], truncation=True, max_length=256)

def main(args):
    data_files = {"train": args.train_csv}
    if args.valid_csv:
        data_files["validation"] = args.valid_csv
    ds = load_dataset("csv", data_files=data_files)
    # ensure label ints
    def cast_label(example):
        example["label"] = int(example["label"])
        return example
    ds = ds.map(cast_label)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized = ds.map(lambda ex: preprocess_function(ex, tokenizer), batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    data_collator = DataCollatorWithPadding(tokenizer)
    metric = evaluate.load("roc_auc")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = logits[:, 1]
        return {"roc_auc": float(metric.compute(predictions=probs, references=labels)["roc_auc"])}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print("Saved model to", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-csv", required=False)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="model/text_distilbert")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    args = parser.parse_args()
    main(args)