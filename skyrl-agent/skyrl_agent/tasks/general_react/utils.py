from typing import Dict, Any, Optional, List, Union
import asyncio
import re

from skyrl_agent.tasks.base import BaseTask
from skyrl_agent.dispatcher.async_utils import call_sync_from_async


def _extract_question_from_instance(instance: Dict[str, Any]) -> str:
    """Extract the question from an instance, handling multiple prompt formats.

    Handles the following cases:
    - raw_prompt field (training mode)
    - prompt field (inference mode)
    - numpy array wrapped prompts
    - Various question prefixes like "Answer the given question:", "Question:", etc.

    Args:
        instance: The instance dictionary containing the prompt

    Returns:
        The extracted question text
    """
    # Try to get the prompt content
    prompt_content = ""

    if "raw_prompt" in instance:
        raw_prompt = instance["raw_prompt"]
        # Handle numpy array
        if hasattr(raw_prompt, "tolist"):
            raw_prompt = raw_prompt.tolist()
        if isinstance(raw_prompt, list) and len(raw_prompt) > 0:
            if isinstance(raw_prompt[0], dict):
                prompt_content = raw_prompt[0].get("content", "")
            else:
                prompt_content = str(raw_prompt[0])
    elif "prompt" in instance:
        prompt = instance["prompt"]
        if isinstance(prompt, list) and len(prompt) > 0:
            if isinstance(prompt[0], dict):
                prompt_content = prompt[0].get("content", "")
            else:
                prompt_content = str(prompt[0])
        elif isinstance(prompt, str):
            prompt_content = prompt

    # Also check for explicit question field
    if "question" in instance:
        return str(instance["question"])

    # Remove common prefixes to extract the actual question
    prefixes_to_remove = [
        "Answer the given question:",
        "Answer the following question:",
        "Question:",
        "Please answer:",
        "Answer:",
    ]

    question = prompt_content
    for prefix in prefixes_to_remove:
        question = question.replace(prefix, "")

    # Clean up whitespace
    question = question.strip()

    return question


class GeneralReactTask(BaseTask):
    @classmethod
    async def initialize_runtime(cls):
        pass

    @classmethod
    def get_instruction(cls, instance: Dict[str, Any]) -> str:
        print(instance)
        # (TODO) Dacheng: A hack to make inference only compatible.
        # During inference, the key is "prompt"
        # During training, the key is "raw_prompt"
        if "raw_prompt" in instance:
            import numpy

            assert isinstance(
                instance.get("raw_prompt"), numpy.ndarray
            ), f"Raw prompt must be a numpy array, but got {type(instance.get('raw_prompt'))}"
            assert (
                len(instance.get("raw_prompt")) == 1
            ), f"Raw prompt must have only one element, but got {len(instance.get('raw_prompt'))}"
            prompt = list(instance.get("raw_prompt"))
        else:
            prompt = instance.get("prompt")

        print(f"Prompt: {prompt}")

        # assume prompt is a list of messages
        assert isinstance(prompt, list), f"Prompt must be a list, but got {type(prompt)}"

        # Check if there's already a system message
        has_system = any(msg.get("role") == "system" for msg in prompt)
        data_source = instance.get("data_source", "")

        if not has_system:
            # Prepend a dummy system prompt so that the agent can use tools by the convert_fncall_messages_to_non_fncall_messages function.
            # From Retool prompt: https://arxiv.org/pdf/2504.11536
            if data_source.startswith("math"):
                system_prompt = {
                    "role": "system",
                    "content": "Please solve the problem with the following tools and return the final answer inside the finish tool. \
                    If there are additional requirments such as the answer should be included inside \\boxed{}, please return the answer in the format of \
                    <function=finish> \
                    <parameter=answer>\\boxed{'The final answer goes here.'}</parameter> \
                    </function>",
                }
            elif data_source.startswith("codegen"):
                system_prompt = {
                    "role": "system",
                    "content": "Please solve the problem with the following tools and return the final answer inside the finish tool. \
                    Given a coding task, you can use the code interpreter tool to run code to test your solution. \
                    Please return the final code solution in the format of \
                    <function=finish> \
                    <parameter=answer>```python\n# The final answer goes here.\n```</parameter> \
                    </function>",
                }
            elif data_source in ["2wikimultihopqa", "bamboogle", "hotpotqa", "musique", "nq", "popqa", "triviaqa"]:
                system_prompt = {
                    "role": "system",
                    "content": "Please solve the problem with the following tools AS MUCH AS POSSIBLE and return the final answer inside the finish tool. \
                    Please return the final answer in the format of \
                    <function=finish> \
                    <parameter=answer>The final answer goes here.</parameter> \
                    </function>",
                }
            elif data_source.startswith("browsecomp"):
                system_prompt = {
                    "role": "system",
                    "content": "Please solve the problem with the following tools and return the final answer inside the finish tool. You need to follow these steps: \
                    1. Analyze the problem carefully and break it down into smaller parts. \
                    2. Use the available tools to gather information and answer sub-questions as needed. \
                    3. Synthesize the information you have gathered to arrive at the final answer. For each conclusion you reach, make sure to provide evidence from the gathered information. \
                    4. Return the final answer in the format of \
                    <function=finish> \
                    <parameter=answer>The final answer goes here.</parameter> \
                    </function>",
                }
            elif data_source.startswith("ruler"):
                system_prompt = {
                    "role": "system",
                    "content": "When you find the answer (or finish reading all chunks), provide your final answer using:\
                    \n<function=finish>\n<parameter=answer>The final answer goes here.</parameter>\n</function>",
                }
            else:
                assert False, f"Data source {data_source} is not supported for ReAct agent."
            prompt = [system_prompt] + prompt

        print(f"Prompt after system prompt: {prompt}")
        return prompt

    @classmethod
    def complete_runtime(cls):
        pass

    @classmethod
    async def evaluate_result(
        cls, result: any, instance: any, data_source: str, instance_id: int, trajectory_id: int
    ) -> float:
        # print(f"Evaluating result: {result=} {instance=} {data_source=} {instance_id=} {trajectory_id=}")
        ground_truth = instance["reward_model"]["ground_truth"]
        extra_info = instance["extra_info"]
        print(f"Evaluating result: {result=}")
        if not result:
            return 0.0
        if data_source == "ToRL":
            from skyrl_agent.tasks.verifiers import torl

            return torl.compute_score(result, ground_truth)
        elif data_source.startswith("math"):
            from skyrl_agent.tasks.verifiers import naive_dapo

            print(f"Evaluating math task with data_source: {data_source}, got {result=} {ground_truth=} {extra_info=}")
            res = naive_dapo.compute_score(result, ground_truth, extra_info=extra_info)
            print(f"Evaluated math task with data_source: {data_source}, got {res=}")
            return res["score"]
        # code generation
        elif data_source.startswith("codegen"):
            from skyrl_agent.tasks.verifiers import coder1

            # print(f"Getting result {result}")
            print(f"Evaluating codegen task with data_source: {data_source}, got {result=}")
            res = coder1.compute_score(result, ground_truth, extra_info=extra_info)
            print(f"Evaluated codegen task with data_source: {data_source}, got {res=}")
            # print(f"Evaluating codegen task with data_source: {data_source}, got {score=} {extracted_model_output=}")
            print(f"Evaluating codegen task with data_source: {data_source}, got {res['score']=}")
            return res["score"]
        elif data_source in ["2wikimultihopqa", "bamboogle", "hotpotqa", "musique", "nq", "popqa", "triviaqa"]:
            ground_truth = instance["reward_model"]["ground_truth"]
            from skyrl_agent.tasks.verifiers import qa

            print(f"Evaluating nq / hotpotqa like task with data_source: {data_source}, got {result=}")
            res = qa.compute_score_em(result, ground_truth)
            print(f"Evaluated nq / hotpotqa like task with data_source: {data_source}, got {res=}")
            return res["score"]
        elif data_source.startswith("browsecomp"):
            ground_truth = instance["reward_model"]["ground_truth"]
            from skyrl_agent.tasks.verifiers import qa

            print(f"Evaluating {data_source} task with data_source: {data_source}, got {result=}")
            # Extract question from prompt, handling multiple formats
            question = _extract_question_from_instance(instance)
            print(f"during evaluation, Question: {question}")
            res = await call_sync_from_async(qa.compute_score_browsecomp, result, ground_truth, question)
            print(f"Evaluated {data_source} task with data_source: {data_source}, got {res=}")
            return res["score"]
        elif data_source.startswith("ruler"):
            from skyrl_agent.tasks.verifiers import qa

            ground_truth = instance["reward_model"]["ground_truth"]
            print(f"Evaluating ruler task with data_source: {data_source}, got {result=}")
            # Extract question from prompt, handling multiple formats
            question = _extract_question_from_instance(instance)
            print(f"Question: {question}")
            res = await call_sync_from_async(qa.compute_score_ruler, result, ground_truth, question)
            print(f"Evaluated ruler task with data_source: {data_source}, got {res=}")
            return res["score"]
        elif data_source.startswith("sql"):
            # SQL tasks - use exact match or execution-based evaluation
            from skyrl_agent.tasks.verifiers import qa
            print(f"Evaluating SQL task with data_source: {data_source}, got {result=}")
            res = qa.compute_score_em(result, ground_truth)
            print(f"Evaluated SQL task with data_source: {data_source}, got {res=}")
            return res["score"]
        elif data_source.startswith("tool") or data_source.startswith("agent"):
            # Generic tool/agent tasks - use exact match
            from skyrl_agent.tasks.verifiers import qa
            print(f"Evaluating tool/agent task with data_source: {data_source}, got {result=}")
            res = qa.compute_score_em(result, ground_truth)
            print(f"Evaluated tool/agent task with data_source: {data_source}, got {res=}")
            return res["score"]
        elif data_source.startswith("reasoning") or data_source.startswith("logic"):
            # Reasoning/logic tasks - use math verifier for structured output
            from skyrl_agent.tasks.verifiers import naive_dapo
            print(f"Evaluating reasoning task with data_source: {data_source}, got {result=}")
            res = naive_dapo.compute_score(result, ground_truth, extra_info=extra_info)
            print(f"Evaluated reasoning task with data_source: {data_source}, got {res=}")
            return res["score"]
        else:
            # Fallback: Generic evaluation using exact match or fuzzy matching
            from skyrl_agent.tasks.verifiers import qa
            print(f"WARNING: Using generic fallback evaluator for unknown data_source: {data_source}")
            print(f"Evaluating with generic fallback: {result=} {ground_truth=}")

            # Try exact match first
            res = qa.compute_score_em(result, ground_truth)
            if res["score"] > 0:
                print(f"Generic fallback evaluation (exact match) for {data_source}: {res=}")
                return res["score"]

            # If no exact match, try normalized string comparison
            if result and ground_truth:
                result_norm = str(result).strip().lower()
                gt_norm = str(ground_truth).strip().lower()
                if result_norm == gt_norm:
                    print(f"Generic fallback evaluation (normalized match) for {data_source}: score=1.0")
                    return 1.0
                # Check if result contains ground truth or vice versa
                if gt_norm in result_norm or result_norm in gt_norm:
                    print(f"Generic fallback evaluation (partial match) for {data_source}: score=0.5")
                    return 0.5

            print(f"Generic fallback evaluation (no match) for {data_source}: score=0.0")
            return 0.0

        if isinstance(res, dict):
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])


class DummyReactTask(GeneralReactTask):
    # For Dummy ReAct Agent that does not use tools
    @classmethod
    def get_instruction(cls, instance: Dict[str, Any]) -> str:
        prompt = instance.get("prompt")
        return prompt


if __name__ == "__main__":
    # Simple test
    # ToRL
    solution_str = "\\boxed{1}"
    ground_truth = "1"
    instance = {"reward_model": {"ground_truth": ground_truth}, "extra_info": {}}
    print(asyncio.run(GeneralReactTask.evaluate_result(solution_str, instance, "math", 0, 0)))
