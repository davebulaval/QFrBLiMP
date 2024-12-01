from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from transformers import logging
from unsloth import FastLanguageModel
from baseline_models import RandomClassBaselineModel

logging.set_verbosity_warning()


def model_tokenizer_factory(
    model_name, device, token, seed: int = 42, class_to_predict: int = 0
):
    if "gpt" in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    elif "bert" in model_name.lower():
        # BERT based model (along with RoBERTa).
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif "aléatoire" in model_name.lower():
        model = RandomClassBaselineModel(seed=seed)
        tokenizer = None
    elif "bloom" in model_name.lower() or "moe" in model_name.lower():
        bnb_configs = BitsAndBytesConfig(load_in_8bit=True, low_cpu_mem_usage=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            quantization_config=bnb_configs,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            token=token,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    return model, tokenizer
