import argparse
import json

from model_size import model_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name to process the evaluation loop on.",
    )

    args = parser.parse_args()

    model_name = args.model_name

    num_params = model_size(model_name)

    # We load the file to append the new datapoint to it.
    with open("models_size.json", "r", encoding="utf-8") as file:
        models_size = json.load(file)

    models_size.update({model_name: num_params})

    # We dump the models_size
    with open("models_size.json", "w", encoding="utf-8") as file:
        json.dump(models_size, file, ensure_ascii=False)
