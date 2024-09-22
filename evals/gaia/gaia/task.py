from pathlib import Path
from typing import Callable, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import Solver, basic_agent, system_message
from inspect_ai.tool import bash, web_search

from .dataset import gaia_dataset

COMPOSE_FILE = Path(__file__).parent / "compose.yaml"


@task
def gaia(
    plan: Solver | None = None,
    input_prompt: str | None = None,
    max_messages: int = 30,
    subset: Literal[
        "2023_all", "2023_level1", "2023_level2", "2023_level3"
    ] = "2023_all",
    split: Literal["test", "validation"] = "validation",
    filter: Callable[[Sample], bool] | None = None,
) -> Task:
    # read dataset
    dataset = gaia_dataset(
        input_prompt=input_prompt or DEFAULT_INPUT_PROMPT,
        subset=subset,
        split=split,
        filter=filter if filter else lambda x: True,
    )

    # provide default plan if required
    if plan is None:
        plan = basic_agent(
            init=system_message(
                "Please solve the coding task below. Once you are done, "
                + "use the submit() tool to provide your answer."
            ),
            tools=[bash(), web_search()],
        )

    # resolve scorer (test split has no answers)
    scorer = match() if split == "validation" else None

    # return task
    return Task(
        dataset=dataset,
        plan=plan,
        scorer=scorer,
        sandbox=("docker", COMPOSE_FILE.as_posix()),
        max_messages=max_messages,
    )


DEFAULT_INPUT_PROMPT = """Please answer the question below. You should:

- Return only your answer, which should be a number, or a short phrase with as
  few words as possible, or a comma separated list of numbers and/or strings.
- If the answer is a number, return only the number without any units unless
  specified otherwise.
- If the answer is a string, don't include articles, and don't use
  abbreviations (e.g. for states).
- If the answer is a comma separated list, apply the above rules to each
  element in the list.

Any files or attachments mentioned in the question can be found in the
/shared_files/ directory (some questions do not have associated files). Here
is the question:

{question}"""
