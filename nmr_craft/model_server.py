#!/usr/bin/env python3
"""Persistent Qwen service exposing inverse and forward NMR endpoints."""
from __future__ import annotations

import argparse
import json
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import forward_prompt, inverse_prompt, spectrum_text


def normalize_smiles(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return re.sub(r"^<SMILES>\s*", "", first_line).strip()


class QwenService:
    def __init__(
        self,
        model_path: str,
        role: str,
        dtype: str,
        max_input_length: int,
        prompt_format: str,
        forward_batch_size: int,
    ) -> None:
        if forward_batch_size < 1:
            raise ValueError("forward_batch_size must be positive")
        self.model_path = model_path
        self.role = role
        self.max_input_length = max_input_length
        self.prompt_format = prompt_format
        self.forward_batch_size = forward_batch_size
        self.lock = threading.Lock()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.model.eval()
        self.device = self.model.device

    def chat_prompt(self, content: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def inverse(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.role not in {"inverse", "both"}:
            raise ValueError("inverse capability is disabled")

        count = int(request.get("num_candidates", 20))
        mode = request.get("decode_mode", "beam")
        if not 1 <= count <= 128:
            raise ValueError("num_candidates must be in [1, 128]")

        content = inverse_prompt(
            request.get("h_shift", []),
            request.get("c_shift", []),
            request.get("formula"),
            request.get("h_peaks"),
            self.prompt_format,
        )
        prompt = self.chat_prompt(content)
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)
        generation = {
            **inputs,
            "max_new_tokens": int(request.get("max_new_tokens", 256)),
            "num_return_sequences": count,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if mode == "beam":
            generation.update(
                num_beams=count,
                do_sample=False,
                early_stopping=True,
            )
        elif mode == "sample":
            generation.update(
                num_beams=1,
                do_sample=True,
                temperature=float(request.get("temperature", 1.0)),
                top_p=float(request.get("top_p", 1.0)),
                top_k=int(request.get("top_k", 0)),
            )
        else:
            raise ValueError("decode_mode must be beam or sample")

        with self.lock, torch.inference_mode():
            generated = self.model.generate(**generation)

        prompt_length = inputs["input_ids"].shape[1]
        sequences = generated.sequences
        candidates = [
            normalize_smiles(
                self.tokenizer.decode(
                    sequence[prompt_length:], skip_special_tokens=True
                )
            )
            for sequence in sequences
        ]
        transition = self.model.compute_transition_scores(
            sequences,
            generated.scores,
            beam_indices=getattr(generated, "beam_indices", None),
            normalize_logits=True,
        )
        candidate_scores = []
        for token_scores in transition:
            finite = token_scores[torch.isfinite(token_scores)]
            total = float(finite.sum().item()) if finite.numel() else float("-inf")
            tokens = int(finite.numel())
            candidate_scores.append(
                {
                    "mean_logprob": total / tokens if tokens else float("-inf"),
                    "sum_logprob": total,
                    "token_count": tokens,
                }
            )
        return {
            "candidates": candidates,
            "candidate_scores": candidate_scores,
            "decode_mode": mode,
            "num_candidates": count,
        }

    def _forward_features(
        self,
        smiles: str,
        h_shift: list[float],
        c_shift: list[float],
    ) -> dict[str, torch.Tensor]:
        prompt_text = forward_prompt(smiles)
        prompt = self.chat_prompt(prompt_text)
        full = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": spectrum_text(h_shift, c_shift)},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_length = len(
            self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
        encoded = self.tokenizer(
            full,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_input_length,
        )
        labels = list(encoded["input_ids"])
        labels[: min(prompt_length, len(labels))] = [-100] * min(
            prompt_length, len(labels)
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"]),
            "attention_mask": torch.tensor(encoded["attention_mask"]),
            "labels": torch.tensor(labels),
        }

    def _score_forward_batch(
        self,
        candidates: list[str],
        h_shift: list[float],
        c_shift: list[float],
    ) -> list[dict[str, Any]]:
        features = [
            self._forward_features(smiles, h_shift, c_shift)
            for smiles in candidates
        ]
        width = max(item["input_ids"].numel() for item in features)
        padded: dict[str, list[torch.Tensor]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for item in features:
            padding = width - item["input_ids"].numel()
            padded["input_ids"].append(
                torch.cat(
                    [
                        item["input_ids"],
                        torch.full((padding,), self.tokenizer.pad_token_id),
                    ]
                )
            )
            padded["attention_mask"].append(
                torch.cat(
                    [
                        item["attention_mask"],
                        torch.zeros(padding, dtype=torch.long),
                    ]
                )
            )
            padded["labels"].append(
                torch.cat([item["labels"], torch.full((padding,), -100)])
            )
        tensors = {
            key: torch.stack(values).long().to(self.device)
            for key, values in padded.items()
        }
        with self.lock, torch.inference_mode():
            logits = self.model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
            ).logits

        shifted_logits = logits[:, :-1, :]
        shifted_labels = tensors["labels"][:, 1:]
        mask = shifted_labels.ne(-100)
        safe_labels = shifted_labels.masked_fill(~mask, 0)
        logprobs = (
            F.log_softmax(shifted_logits, dim=-1)
            .gather(-1, safe_labels.unsqueeze(-1))
            .squeeze(-1)
            .masked_fill(~mask, 0.0)
        )
        scores = []
        for index, smiles in enumerate(candidates):
            tokens = int(mask[index].sum())
            total = float(logprobs[index].sum())
            mean = total / tokens if tokens else float("-inf")
            scores.append(
                {
                    "candidate_smiles": smiles,
                    "token_count": tokens,
                    "sum_logprob": total,
                    "mean_logprob": mean,
                    "nll": -mean,
                    "perplexity": (
                        math.exp(-mean) if math.isfinite(mean) else None
                    ),
                }
            )
        return scores

    def forward(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.role not in {"forward", "both"}:
            raise ValueError("forward capability is disabled")
        candidates = [
            str(value)
            for value in (
                request.get("candidates") or [request.get("smiles")]
            )
            if value
        ]
        if not 1 <= len(candidates) <= 256:
            raise ValueError("candidates must contain 1..256 SMILES")

        h_shift = request.get("h_shift", [])
        c_shift = request.get("c_shift", [])
        scores = []
        for start in range(0, len(candidates), self.forward_batch_size):
            scores.extend(
                self._score_forward_batch(
                    candidates[start : start + self.forward_batch_size],
                    h_shift,
                    c_shift,
                )
            )
        return {"scores": scores}


class Handler(BaseHTTPRequestHandler):
    service: QwenService

    def reply(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self.reply(404, {"error": "not found"})
            return
        self.reply(
            200,
            {
                "status": "ok",
                "role": self.service.role,
                "model": Path(self.service.model_path).name,
                "prompt_format": self.service.prompt_format,
                "forward_batch_size": self.service.forward_batch_size,
            },
        )

    def do_POST(self) -> None:
        try:
            request = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
            )
            if self.path == "/inverse":
                output = self.service.inverse(request)
            elif self.path == "/forward":
                output = self.service.forward(request)
            else:
                self.reply(404, {"error": "not found"})
                return
            self.reply(200, output)
        except Exception as exc:
            self.reply(400, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, message: str, *args: Any) -> None:
        print(message % args, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--role", choices=("forward", "inverse", "both"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--prompt-format", choices=("nmrgym", "nmrtrans"), default="nmrgym")
    parser.add_argument("--forward-batch-size", type=int, default=16)
    args = parser.parse_args()

    service = QwenService(
        args.model_path,
        args.role,
        args.dtype,
        args.max_input_length,
        args.prompt_format,
        args.forward_batch_size,
    )
    Handler.service = service
    print(
        json.dumps(
            {
                "status": "ready",
                "role": args.role,
                "prompt_format": args.prompt_format,
                "forward_batch_size": args.forward_batch_size,
                "port": args.port,
            }
        ),
        flush=True,
    )
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
