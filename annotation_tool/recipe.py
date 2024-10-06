import random
from pathlib import Path

from prodigy.components.loaders import JSONL
from prodigy.core import Arg, recipe
from prodigy.util import set_hashes

random.seed(42)

root = Path(__file__).parent
options = [
    {
        "id": "1",
        "text": "La phrase numéro 1",
    },
    {
        "id": "2",
        "text": "La phrase numéro 2",
    },
]


# Helper functions for adding user provided labels to annotation tasks.
def add_label_options_to_stream(stream):
    for task in stream:
        task["options"] = options

        grammatical = task["grammatical"]
        ungrammatical = task["ungrammatical"]

        # We randomly select the sentence_1 to be selected amongst the two sentences
        choices = [grammatical, ungrammatical]
        sentence_1 = random.choice(choices)
        sentence_2 = [choice for choice in choices if choice != sentence_1]

        task["sentence_1"] = sentence_1
        task["sentence_2"] = sentence_2
        yield task


@recipe(
    "fr_blimp",
    dataset=Arg(help="Dataset to save answers to."),
    source=("The source data as a JSONL file", "positional", None, str),
)
def fr_blimp(dataset: str, source: str):
    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    stream = list(
        set_hashes(eg, input_keys=("grammatical", "ungrammatical")) for eg in stream
    )

    # Randomized the stream
    # We need to do it before add_label_options_to_stream since it return a generator
    random.shuffle(stream)

    # Add labels to each task in stream
    stream = add_label_options_to_stream(stream)

    blocks = [
        {"view_id": "html", "html_template": (root / "recipe.html").read_text()},
        {
            "view_id": "choice",
            "text": None,
            "field_id": "choix_phrase",
        },
    ]

    return {
        "view_id": "blocks",
        "dataset": dataset,  # save annotations in this dataset
        "stream": stream,
        "config": {  # Additional config settings, mostly for app UI
            "exclude_by": "input",  # Hash value used to filter out already seen examples
            "blocks": blocks,
        },
    }
