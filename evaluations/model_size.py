import torch
from dotenv import dotenv_values

from model_tokenizer_factory import model_tokenizer_factory


def model_size(model_name: str):
    secrets = dotenv_values(".env")

    huggingface_token = secrets["huggingface_token"]

    device = torch.device("cuda")

    model, _ = model_tokenizer_factory(
        # To clean model name when we have applied a '_prompting' to it.
        model_name=(
            model_name
            if "_prompting" not in model_name
            else model_name.replace("_prompting", "")
        ),
        device=device,
        token=huggingface_token,
    )

    num_params = model.num_parameters()

    return num_params
