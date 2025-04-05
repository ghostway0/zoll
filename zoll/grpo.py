from typing import List, Any, Callable, Dict, Union
from dataclasses import dataclass
from collections import defaultdict

from torch.nn.utils.rnn import pad_sequence
import torch
from transformers import Trainer

from zoll.llm import LLMClient, SamplingParams
from zoll.multiturn import run_conversations, ToolEnv, Result


def pad_right(
    tensors: List[torch.Tensor], padding_value: Any, batch_first: bool = True
) -> torch.Tensor:
    return pad_sequence(tensors, batch_first=batch_first, padding_value=padding_value)


def upload_to_inference(self, client: LLMClient):
    for name, param in self.model.named_parameters():
        # sharding context may gather here

        client.load_weight(name, param.data)


@dataclass
class GRPOParams:
    group_size: int
    kl_coeff: float
    max_steps: int


RewardFunc = Callable[[Result], float]


class GRPOTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel,
        llm_client: LLMClient,
        reward_func: RewardFunc,
        train_dataset: Union[Dataset, IterableDataset],
        callbacks: List[TrainerCallback],
        grpo_params: GRPOParams,
        env: ToolEnv,
        sampling_params: SamplingParams,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
    ):
        def data_collator(features):
            return features

        processing_class = AutoTokenizer.from_pretrained(
            model.config._name_or_path, padding_side="left"
        )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.reward_func = reward_func
        self.grpo_params = grpo_params
        self.sampling_params = sampling_params
        self.llm_client = llm_client
        self.env = env

        self.pad_token_id = (processing_class.pad_token_id,)
        self.bos_token_id = (processing_class.bos_token_id,)
        self.eos_token_id = (processing_class.eos_token_id,)

    def _generate_and_score(self, inputs: List[str]):
        results = run_conversations(
            self.env,
            self.llm_client,
            inputs,
            self.grpo_params.group_size,
            self.sampling_params,
            self.grpo_params.max_steps,
        )

        rewards = torch.empty((1, len(results)))

        grouped_results: Dict[str, List[Result]] = defaultdict(list)
        for res in results:
            grouped_results[res.initial_prompt].append(res)

        rewards = torch.zeros(len(results))
        result_to_idx = {id(res): i for i, res in enumerate(results)}
        for prompt, group in grouped_results.items():
            for res in group:
                rewards[result_to_idx[id(res)]] = self.reward_func(res)

        num_prompts = len(inputs)
        rewards = rewards.view(num_prompts, self.grpo_params.group_size)

        mean_rewards = rewards.mean(dim=1, keepdim=True)  # [num_prompts, 1]
        std_rewards = rewards.std(dim=1, keepdim=True)  # [num_prompts, 1]
        advantages = rewards - mean_rewards  # [num_prompts, group_size]

        prompt_inputs = self.processing_class(
            text=inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # I think this is wrong. it needs to be compatible with completion_ids, which we should be generating from tokenizing
        mean_rewards = mean_rewards.repeat(1, self.grpo_params.group_size).view(-1)
        std_rewards = std_rewards.repeat(1, self.grpo_params.group_size).view(-1)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }
