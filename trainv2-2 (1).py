import os
import pandas as pd
import torch
import wandb

# Check GPU availability
print("CUDA Available:", torch.cuda.is_available())

import logging
import pandas as pd
import librosa
import numpy as np
import torch
import wandb
from argparse import ArgumentParser
from pathlib import Path
from transformers import (
    EvalPrediction,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
import evaluate

class ReadSpeechDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,  # Expecting a DataFrame
        do_awgn: bool = False,
        awgn_snr: float = 0.05,
        do_masking: bool = False,
        max_mask_len: int = 3200,
    ) -> None:
        self.df = df  # Store DataFrame
        self.do_awgn = do_awgn
        self.awgn_snr = awgn_snr
        self.do_masking = do_masking
        self.max_mask_len = max_mask_len

    def _awgn(self, audio: np.ndarray) -> np.ndarray:
        noise = np.random.randn(len(audio))
        audio_power = np.sum(audio**2)
        noise_power = np.sum(noise**2)
        scale = np.sqrt(audio_power / noise_power * 10 ** -(self.awgn_snr / 10))
        audio = audio + scale * noise
        audio = np.clip(audio, -1, 1)
        return audio

    def _masking(self, audio: np.ndarray) -> np.ndarray:
        mask_len = np.random.randint(0, self.max_mask_len)
        mask_start = np.random.randint(0, len(audio) - mask_len)
        audio[mask_start : mask_start + mask_len] = 0
        return audio

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        audio_path = self.df.iloc[index]["path"]
        text = self.df.iloc[index]["transcript"]

        # Default silent audio in case of failure
        audio = np.zeros(16000)

        try:
            # Try loading the audio file
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            print(f"[ERROR] Failed to load {audio_path}: {e}")

        # Apply augmentations if needed
        if self.do_awgn:
            audio = self._awgn(audio)
        if self.do_masking:
            audio = self._masking(audio)

        # Ensure valid return
        return {"audio": audio, "text": text}



class Collator:
    def __init__(self, processor: Wav2Vec2Processor) -> None:
        self.processor = processor

    def __call__(self, features: "list[dict[str, np.ndarray]]"):
        for i, x in enumerate(features):
          if "audio" not in x:
            print(f"[DEBUG] Missing 'audio' at index {i}: {x}")
        return self.processor(audio=[x["audio"] for x in features], sampling_rate=16000, text=[x["text"] for x in features])
        features = [
            self.processor(audio=x["audio"], sampling_rate=16000, text=x["text"])
            for x in features
        ]
        input_features = [{"input_values": x["input_values"][0]} for x in features]
        labels = [{"input_ids": x["labels"]} for x in features]

        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        labels_batch = self.processor.pad(
            labels=labels, padding=True, return_tensors="pt"
        )
        labels = labels_batch.input_ids.masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return {"input_values": batch["input_values"], "labels": batch["labels"]}

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred: "EvalPrediction", tokenizer: Wav2Vec2CTCTokenizer):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids)
    label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

def main():
    parser = ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--wandb_proj", type=str)
    parser.add_argument("--resume_from_checkpoint", type=str)
    parser.add_argument("--base_model", type=str, default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument("--do_awgn", action="store_true")
    parser.add_argument("--awgn_snr", type=float, default=0.05)
    parser.add_argument("--do_masking", action="store_true")
    parser.add_argument("--max_mask_len", type=int, default=3200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=300000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    tokenizer = Wav2Vec2CTCTokenizer(vocab_file="/content/vocab.json", word_delimiter_token=" ")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.base_model)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(args.base_model, vocab_size=tokenizer.vocab_size)

    train_dataset = ReadSpeechDataset(train_df, do_awgn=args.do_awgn, do_masking=args.do_masking)
    eval_dataset = ReadSpeechDataset(test_df)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=args.output_dir),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Collator(processor),
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.evaluate()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
