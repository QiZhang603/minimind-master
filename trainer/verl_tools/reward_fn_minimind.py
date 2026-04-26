import os
import re
import threading
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


class RewardModelScorer:
    def __init__(self, model_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype)
        self.model.eval().to(device)

    @torch.no_grad()
    def get_score(self, question: str, answer: str) -> float:
        messages = [{"role": "user", "content": question}]
        score = self.model.get_score(self.tokenizer, messages, answer)
        return float(score)


_REWARD_MODEL = None
_REWARD_MODEL_LOCK = threading.Lock()


def _get_reward_model() -> RewardModelScorer | None:
    global _REWARD_MODEL

    if _REWARD_MODEL is not None:
        return _REWARD_MODEL

    model_path = os.environ.get("MINIMIND_REWARD_MODEL_PATH", "").strip()
    if not model_path:
        return None

    with _REWARD_MODEL_LOCK:
        if _REWARD_MODEL is None:
            _REWARD_MODEL = RewardModelScorer(model_path=model_path)

    return _REWARD_MODEL


def rep_penalty(text: str, n: int = 3, cap: float = 0.5) -> float:
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams))


def _get_question(extra_info: Any) -> str:
    if isinstance(extra_info, dict):
        question = extra_info.get("question", "")
        if question:
            return str(question)

        prompt = extra_info.get("prompt")
        if isinstance(prompt, list):
            for msg in reversed(prompt):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return str(msg.get("content", ""))

    return ""


def compute_score_minimind(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict[str, float]:
    del data_source, ground_truth, kwargs

    response = str(solution_str or "")
    reward = 0.0

    reward += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

    answer = response
    if "</think>" in response:
        thinking_content, answer_content = response.split("</think>", 1)
        reward += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
        reward += 0.25 if response.count("</think>") == 1 else -0.25
        answer = answer_content.strip()

    reward -= rep_penalty(answer)

    rm_score = 0.0
    question = _get_question(extra_info)
    scorer = _get_reward_model()
    if scorer is not None and question and answer:
        try:
            rm_score = scorer.get_score(question=question, answer=answer)
        except Exception:
            rm_score = 0.0

    total = reward + rm_score
    return {
        "score": float(total),
        "rule_score": float(reward),
        "rm_score": float(rm_score),
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict[str, float]:
    return compute_score_minimind(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
