import os
import shutil
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import snapshot_download  # type: ignore
from platformdirs import user_cache_dir

from inspect_ai.dataset import Dataset, Sample, hf_dataset


def gaia_dataset(
    input_prompt: str,
    subset: str,
    split: str,
    filter: Callable[[Sample], bool] = lambda x: True,
) -> Dataset:
    # use user cache dir for dataset
    GAIA_DATASET_LOCATION = Path(user_cache_dir("gaia_eval")) / "GAIA"

    shutil.rmtree(GAIA_DATASET_LOCATION, True)

    # download dataset if required
    if not os.path.exists(GAIA_DATASET_LOCATION):
        GAIA_DATASET_LOCATION.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id="gaia-benchmark/GAIA",
                repo_type="dataset",
                local_dir=GAIA_DATASET_LOCATION,
            )
        except Exception as ex:
            shutil.rmtree(GAIA_DATASET_LOCATION, True)
            raise ex

    # map record to sample
    def record_to_sample(record: dict[str, Any]) -> Sample:
        # map fields
        sample = Sample(
            input=input_prompt.format(question=record["Question"]),
            target=record["Final answer"],
            id=record["task_id"],
            metadata={
                "level": record["Level"],
                "Annotator Metadata": record["Annotator Metadata"],
            },
            setup="mkdir -p /shared_files/",
        )

        # apply input prompt
        sample.input = input_prompt.format(question=sample.input)

        # provide sample files
        files_location = GAIA_DATASET_LOCATION / "2023" / split
        files = [file for file in os.listdir(files_location) if str(sample.id) in file]
        if len(files) > 0:
            sample.files = {
                "/shared_files/" + files[0]: (files_location / files[0]).as_posix()
            }

        return sample

    # read dataset
    print(GAIA_DATASET_LOCATION)
    dataset = hf_dataset(
        GAIA_DATASET_LOCATION.as_posix(),
        name=subset,
        split=split,
        sample_fields=record_to_sample,
        trust=True,  # Trust GAIA's remote code execution during dataset loading
    )

    # apply filter (if any) and return
    return dataset.filter(filter)
