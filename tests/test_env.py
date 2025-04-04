from typing import Optional, List, Dict

import torch
import pytest

from zoll.llm import LLMClient, SamplingParams
from zoll.multiturn import (
    ToolEnv,
    ToolInfo,
    ToolCall,
    ToolError,
    Answer,
    Result,
    Conversation,
    TOOL_PROMPT_TEMPLATE,
    TOOL_CALL_REGEX,
    ANSWER_REGEX,
    _run_single_conversation,
    run_conversations,
)


class MockLLMClient(LLMClient):
    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses if responses else []
        self.call_count = 0
        self.requests_log: List[Dict] = []

    def generate(self, messages: List[Dict[str, str]], params: SamplingParams) -> str:
        self.requests_log.append({"messages": messages, "params": params.dict()})
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        raise ValueError("MockLLMClient ran out of responses")

    def load_weight(self, name: str, weight: torch.Tensor):
        print(f"Mock load_weight called with name: {name}")

    def close(self):
        print("Mock close called")


@pytest.fixture
def sampling_params():
    return SamplingParams(temperature=0.1, max_tokens=100)


@pytest.fixture
def empty_tool_env():
    return ToolEnv()


@pytest.fixture
def basic_tool_env():
    env = ToolEnv()

    def add(a: int, b: int) -> int:
        return a + b

    def search(query: str) -> str:
        return f"Search results for '{query}'"

    def fail_tool():
        raise ValueError("Tool failed intentionally")

    env.add_tool(add, description="Adds two numbers.")
    env.add_tool(search, description="Searches the web.")
    env.add_tool(fail_tool, description="A tool designed to fail.")
    return env


def test_conv_init():
    initial = "Hello"
    system = "You are a bot."
    conv = Conversation(initial_prompt=initial, system_prompt=system)
    assert len(conv.messages) == 2
    assert conv.messages[0] == {"role": "system", "content": system}
    assert conv.messages[1] == {"role": "user", "content": initial}


def test_conv_add_message():
    conv = Conversation("Hi", "Sys")
    conv.add_message("assistant", "How can I help?")
    conv.add_message("user", "Question?")
    assert len(conv.messages) == 4
    assert conv.messages[2] == {"role": "assistant", "content": "How can I help?"}
    assert conv.messages[3] == {"role": "user", "content": "Question?"}


def test_env_add_tool(empty_tool_env):
    env = empty_tool_env

    def my_tool():
        pass

    env.add_tool(my_tool, description="Does something.")
    assert "my_tool" in env.tools
    assert env.tools["my_tool"].name == "my_tool"
    assert env.tools["my_tool"].description == "Does something."
    assert env.tools["my_tool"].func == my_tool


def test_env_dup_tool(empty_tool_env):
    env = empty_tool_env

    def my_tool():
        pass

    env.add_tool(my_tool, description="Desc 1")
    with pytest.raises(ValueError, match="Tool 'my_tool' already exists."):
        env.add_tool(my_tool, description="Desc 2")


def test_env_tool_no_description(empty_tool_env):
    env = empty_tool_env

    def my_tool():
        pass

    with pytest.raises(ValueError, match="Tool 'my_tool' must have a description."):
        env.add_tool(my_tool, description="")


@pytest.mark.parametrize(
    "response, expected",
    [
        (
            '<tool>{"name": "add", "args": {"a": 1, "b": 2}}</tool>',
            ToolCall(name="add", args={"a": 1, "b": 2}),
        ),
        (
            '<tool> {"name": "search", "args": {"query": "test"}} </tool>',
            ToolCall(name="search", args={"query": "test"}),
        ),
        ("<answer>Final Answer</answer>", Answer(text="Final Answer")),
        ("<answer> Final Answer \n </answer>", Answer(text="Final Answer")),
        (
            '<tool>{"name": "unknown", "args": {}}</tool>',
            ToolError(text="Unknown tool name: unknown"),
        ),
        (
            '<tool>{"tool_name": "add", "args": {}}</tool>',
            ToolError(
                text="Invalid tool JSON structure: {'tool_name': 'add', 'args': {}}"
            ),
        ),
        (
            '<tool>{"name": "add", "arguments": {}}</tool>',
            ToolError(
                text="Invalid tool JSON structure: {'name': 'add', 'arguments': {}}"
            ),
        ),
        (
            "<tool>This is not JSON</tool>",
            ToolError(text="Invalid JSON when tried calling tool: 'This is not JSON'"),
        ),
        (
            '<tool>{"name": "add", "args": {"a": 1, "b": 2}} </tool>',
            ToolCall(name="add", args={"a": 1, "b": 2}),
        ),
        ("Some other text", None),
        ("<result>Some result</result>", None),
    ],
)
def test_env_parse_response(basic_tool_env, response, expected):
    parsed = basic_tool_env._parse_response(response)
    assert parsed == expected


def test_env_execute_tool(basic_tool_env):
    tool_call = ToolCall(name="add", args={"a": 5, "b": 3})
    result = basic_tool_env._execute_tool(tool_call)
    assert result == "8"

    tool_call = ToolCall(name="search", args={"query": "pytest"})
    result = basic_tool_env._execute_tool(tool_call)
    assert result == "Search results for 'pytest'"


def test_env_execute_tool_fail(basic_tool_env):
    tool_call = ToolCall(name="fail_tool", args={})
    result = basic_tool_env._execute_tool(tool_call)
    assert result == "Error executing fail_tool: ValueError"


def test_env_tool_missing_args(basic_tool_env):
    tool_call = ToolCall(name="add", args={"a": 1})
    result = basic_tool_env._execute_tool(tool_call)
    assert "TypeError" in result
    assert "Error executing add" in result


def test_env_step_answer(basic_tool_env, sampling_params):
    env = basic_tool_env
    initial_prompt = "Solve this."
    system_prompt = env.format_system_prompt()
    conversation = Conversation(initial_prompt, system_prompt)
    mock_client = MockLLMClient(responses=["<answer>The final answer is 42.</answer>"])

    result = env.step(conversation, mock_client, sampling_params)

    assert isinstance(result, Result)
    assert result.final_answer == "The final answer is 42."
    assert result.initial_prompt == initial_prompt
    assert len(conversation.messages) == 3
    assert conversation.messages[-1] == {
        "role": "assistant",
        "content": "<answer>The final answer is 42.</answer>",
    }


def test_env_step_tool_call(basic_tool_env, sampling_params):
    env = basic_tool_env
    initial_prompt = "Add 5 and 3."
    system_prompt = env.format_system_prompt()
    conversation = Conversation(initial_prompt, system_prompt)
    tool_call_str = '<tool>{"name": "add", "args": {"a": 5, "b": 3}}</tool>'
    mock_client = MockLLMClient(responses=[tool_call_str])

    result = env.step(conversation, mock_client, sampling_params)

    assert result is None
    assert len(conversation.messages) == 4
    assert conversation.messages[-2] == {"role": "assistant", "content": tool_call_str}
    assert conversation.messages[-1] == {
        "role": "user",
        "content": "<result>8</result>",
    }


def test_env_step_tool_error(basic_tool_env, sampling_params):
    env = basic_tool_env
    initial_prompt = "Use a bad tool."
    system_prompt = env.format_system_prompt()
    conversation = Conversation(initial_prompt, system_prompt)
    tool_error_str = "<tool>Invalid JSON {</tool>"
    mock_client = MockLLMClient(responses=[tool_error_str])

    result = env.step(conversation, mock_client, sampling_params)

    assert result is None
    assert len(conversation.messages) == 4
    assert conversation.messages[-2] == {"role": "assistant", "content": tool_error_str}
    assert "Invalid JSON" in conversation.messages[-1]["content"]
    assert conversation.messages[-1]["role"] == "user"
    assert conversation.messages[-1]["content"].startswith("<error>")
    assert conversation.messages[-1]["content"].endswith("</error>")


def test_env_step_nostep(basic_tool_env, sampling_params):
    env = basic_tool_env
    initial_prompt = "Say something weird."
    system_prompt = env.format_system_prompt()
    conversation = Conversation(initial_prompt, system_prompt)
    misaligned_response = "I am just chatting."
    mock_client = MockLLMClient(responses=[misaligned_response])

    result = env.step(conversation, mock_client, sampling_params)

    assert result is None
    assert len(conversation.messages) == 3
    assert conversation.messages[-1] == {
        "role": "assistant",
        "content": misaligned_response,
    }


def test_run_single_conversation(basic_tool_env, sampling_params):
    env = basic_tool_env
    initial_prompt = "What is 10 + 5?"
    mock_client = MockLLMClient(
        responses=[
            '<tool>{"name": "add", "args": {"a": 10, "b": 5}}</tool>',
            "<answer>The result is 15.</answer>",
        ]
    )
    max_steps = 5

    result = _run_single_conversation(
        env, mock_client, initial_prompt, sampling_params, max_steps
    )

    assert isinstance(result, Result)
    assert result.final_answer == "The result is 15."
    assert result.steps_taken == 2
    assert len(result.messages) == 5
    assert result.messages[0]["role"] == "system"
    assert result.messages[1]["role"] == "user"
    assert result.messages[1]["content"] == initial_prompt
    assert result.messages[2]["role"] == "assistant"
    assert (
        result.messages[2]["content"]
        == '<tool>{"name": "add", "args": {"a": 10, "b": 5}}</tool>'
    )
    assert result.messages[3]["role"] == "user"
    assert result.messages[3]["content"] == "<result>15</result>"
    assert result.messages[4]["role"] == "assistant"
    assert result.messages[4]["content"] == "<answer>The result is 15.</answer>"


def test_run_conversation_max_steps(basic_tool_env, sampling_params):
    env = basic_tool_env
    initial_prompt = "Search loop"
    mock_client = MockLLMClient(
        responses=[
            '<tool>{"name": "search", "args": {"query": "a"}}</tool>',
            '<tool>{"name": "search", "args": {"query": "b"}}</tool>',
            '<tool>{"name": "search", "args": {"query": "c"}}</tool>',
        ]
    )
    max_steps = 2

    result = _run_single_conversation(
        env, mock_client, initial_prompt, sampling_params, max_steps
    )
    assert result is None


def test_run_conversation_tool_fail(basic_tool_env, sampling_params):
    env = basic_tool_env
    initial_prompt = "Make the tool fail"
    mock_client = MockLLMClient(
        responses=[
            '<tool>{"name": "fail_tool", "args": {}}</tool>',
            "<answer>I could not execute the tool.</answer>",
        ]
    )
    max_steps = 5

    result = _run_single_conversation(
        env, mock_client, initial_prompt, sampling_params, max_steps
    )

    assert isinstance(result, Result)
    assert result.final_answer == "I could not execute the tool."
    assert result.steps_taken == 2
    assert len(result.messages) == 5
    assert result.messages[3]["role"] == "user"
    assert (
        result.messages[3]["content"]
        == "<result>Error executing fail_tool: ValueError</result>"
    )


def test_run_conversation_step_fail(basic_tool_env, sampling_params, mocker):
    env = basic_tool_env
    initial_prompt = "Cause an error"
    mock_client = MockLLMClient(responses=[""])
    max_steps = 5

    mocker.patch.object(ToolEnv, "step", side_effect=ValueError("LLM exploded"))

    result = _run_single_conversation(
        env, mock_client, initial_prompt, sampling_params, max_steps
    )

    assert result is None
