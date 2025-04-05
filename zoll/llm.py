from typing import Any, List, Dict
from dataclasses import dataclass, field
import torch


@dataclass
class SamplingParams:
    n: int = 1
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 512
    repetition_penalty: float = 1.0
    stop: List[str] = field(default_factory=list)

    def dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "stop": self.stop,
        }


@dataclass
class ChatResponse:
    text: str
    completion_ids: List[int]
    prompt_ids: List[int]


class LLMClient:
    def generate(
        self, message: List[Dict[str, str]], params: SamplingParams
    ) -> List[ChatResponse]: ...

    def load_weight(self, name: str, weight: torch.Tensor): ...

    def close(self): ...
