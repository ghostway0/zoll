import json
import re
from typing import List, Dict, Callable, Any, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from zoll.llm import SamplingParams, LLMClient

TOOL_PROMPT_TEMPLATE = """\
You have access to the following tools:
{tool_descriptions}

Follow these steps strictly:
1. Reason step-by-step internally.
2. If you need to use a tool, respond with ONLY the tool call JSON within <tool> tags: <tool>{{"name": "tool_name", "args": {{...}}}}</tool>. The tool name must be one of [{tool_names}].
3. After a tool call, you will receive the result in <result> tags in the next user message.
4. When you have the final answer, respond with ONLY the answer text within <answer> tags: <answer>Your final answer.</answer>.
"""

TOOL_CALL_REGEX = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)
ANSWER_REGEX = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass
class ToolInfo:
    name: str
    description: str
    func: Callable


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]


# this is pretty much a hack
@dataclass
class ToolError:
    text: str


@dataclass
class Answer:
    text: str


@dataclass
class Result:
    initial_prompt: str
    messages: List[Dict[str, str]]
    final_answer: Optional[str] = None
    steps_taken: int = 0


@dataclass
class Conversation:
    initial_prompt: str
    system_prompt: str
    messages: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        if not self.messages:
            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.initial_prompt},
            ]

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})


class ToolEnv:
    def __init__(self, prompt_template: str = TOOL_PROMPT_TEMPLATE):
        self.prompt_template = prompt_template
        self.tools: Dict[str, ToolInfo] = {}

    def add_tool(self, func: Callable, description: str, name: Optional[str] = None):
        tool_name = name or func.__name__

        if tool_name in self.tools:
            raise ValueError(f"Tool '{tool_name}' already exists.")
        if not description:
            raise ValueError(f"Tool '{tool_name}' must have a description.")

        self.tools[tool_name] = ToolInfo(
            name=tool_name, description=description, func=func
        )

    def format_system_prompt(self) -> str:
        if len(self.tools) == 0:
            return "You are a helpful assistant."

        descriptions = [f"- {n}: {info.description}" for n, info in self.tools.items()]
        tool_names = list(self.tools.keys())

        return self.prompt_template.format(
            tool_descriptions="\n".join(descriptions),
            tool_names=", ".join(f"'{n}'" for n in tool_names),
        )

    def _execute_tool(self, tool_call: ToolCall):
        tool_info = self.tools[tool_call.name]
        result_str: str

        try:
            result_obj = tool_info.func(**tool_call.args)
            result_str = str(result_obj)
        except Exception as e:
            print(
                f"Error executing tool '{tool_call.name}' with args {tool_call.args}: {e}"
            )
            result_str = f"Error executing {tool_call.name}: {type(e).__name__}"

        return result_str

    def _parse_response(
        self, text: str
    ) -> Optional[Union[ToolError, ToolCall, Answer]]:
        tool_match = TOOL_CALL_REGEX.fullmatch(text)
        answer_match = ANSWER_REGEX.fullmatch(text)

        if answer_match:
            return Answer(text=answer_match.group(1).strip())

        if tool_match:
            try:
                call_dict = json.loads(tool_match.group(1).strip())
            except json.JSONDecodeError:
                return ToolError(
                    text=f"Invalid JSON when tried calling tool: '{tool_match.group(1).strip()}'"
                )

            name = call_dict.get("name")
            args = call_dict.get("args")

            if not isinstance(name, str) or not isinstance(args, dict):
                return ToolError(text=f"Invalid tool JSON structure: {call_dict}")

            if name not in self.tools:
                return ToolError(text=f"Unknown tool name: {name}")

            return ToolCall(name=name, args=args)

    def step(
        self, conversation: Conversation, client: LLMClient, params: SamplingParams
    ) -> Optional[Result]:
        raw_response = client.generate(conversation.messages, params).strip()
        conversation.add_message("assistant", raw_response)

        parsed = self._parse_response(raw_response)

        if isinstance(parsed, Answer):
            return Result(
                initial_prompt=conversation.initial_prompt,
                messages=conversation.messages,
                final_answer=parsed.text,
            )
        elif isinstance(parsed, ToolError):
            conversation.add_message("user", f"<error>{parsed.text}</error>")
        elif isinstance(parsed, ToolCall):
            result = self._execute_tool(parsed)
            conversation.add_message("user", f"<result>{result}</result>")
        # else: the llm is misaligned

        return None


def _run_single_conversation(
    env: ToolEnv,
    client: LLMClient,
    initial_prompt: str,
    sampling_params: SamplingParams,
    max_steps: int,
) -> Optional[Result]:
    system_prompt = env.format_system_prompt()
    conversation = Conversation(initial_prompt, system_prompt)
    result = None

    for step_num in range(max_steps):
        try:
            step_result = env.step(conversation, client, sampling_params)

            if step_result is not None:
                result = step_result
                result.steps_taken = step_num + 1
                break

        except Exception:
            return None

    return result


def run_conversations(
    env: ToolEnv,
    client: LLMClient,
    initial_prompt: str,
    group_size: int,
    sampling_params: SamplingParams,
    max_steps: int,
    max_workers: Optional[int] = None,
) -> List[Result]:
    if max_workers is None:
        max_workers = group_size

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_single_conversation,
                env,
                client,
                initial_prompt,
                sampling_params,
                max_steps,
            )
            for _ in range(group_size)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    return results
