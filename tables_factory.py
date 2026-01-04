import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import wandb
import tikzplotly

root = "results"
saving_dir = "figs"
listdir = os.listdir(root)

with open(os.path.join("datastore", "models_size.json"), "r") as file:
    models_size = json.load(file)

lang = "fr"

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs(f"davebulaval/minimal_pair_analysis_{lang}")

accuracies, accuracies_subcat, model_names, model_sizes = [], [], [], []
model_names_clean = []

skip = ["mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
for run in runs:
    if run.state == "finished":
        model_name = [v for k, v in run.config.items() if k == "model_name"][0]
        if model_name not in skip:
            if "_prompting" not in model_name:
                accuracies.append(
                    [
                        v.get("accuracy")
                        for k, v in run.summary._json_dict.items()
                        if not k.startswith("_")
                    ][0]
                )

                accuracies_subcat.append(
                    [
                        v.get("accuracy_per_subcat")
                        for k, v in run.summary._json_dict.items()
                        if not k.startswith("_")
                    ][0]
                )

                model_names.append(model_name)
                model_name_clean = (
                    r"\texttt{"
                    + model_name.split("/")[-1]
                    .capitalize()
                    .replace("bert", "BERT")
                    .replace("Bert", "BERT")
                    .replace("gpt", "GPT")
                    .replace("french", "French")
                    .replace("-unsloth-bnb-4bit", "")
                    .replace("Gpt", "GPT")
                    .replace("-bnb-4bit", "")
                    .replace("instruct", "it")
                    .replace("Xlm", "XLM")
                    .replace("Flan-t5", "FLAN-T5")
                    .replace("Deepseek", "DeepSeek")
                    .replace("llama", "Llama")
                    .replace("alpaca", "Alpaca")
                    .replace("Qwq", "QwQ")
                    .replace("Smollm", "SmolLM")
                    + r"}"
                )
                model_names_clean.append(model_name_clean)

                model_size = models_size.get(model_name)

                model_sizes.append(model_size)

# Project is specified by <entity/project-name>
runs = api.runs(f"davebulaval/minimal_pair_analysis_multiblimp")

second_set = {}
for run in runs:
    if run.state == "finished":
        model_name = [v for k, v in run.config.items() if k == "model_name"][0]

        accuracies_2 = [
            v.get("accuracy")
            for k, v in run.summary._json_dict.items()
            if not k.startswith("_")
        ][0]

        if "Lucie-7B-Instruct" in model_name and not "human" in model_name:
            model_name = "OpenLLM-France/Lucie-7B-Instruct"
        second_set.update({model_name: accuracies_2})

second_set.update({"Annotateurs": 0})


human_ref = accuracies_subcat[74]
formatted_accuracies_subcat = {str(key): [] for key in range(1, 21)}
formatted_accuracies_subcat_diff = {str(key): [] for key in range(1, 21)}
for row in accuracies_subcat:
    for key, value in sorted(list(row.items())):
        content = formatted_accuracies_subcat.get(key)
        content.append(value)

        human_ref_value = human_ref.get(key)

        content_2 = formatted_accuracies_subcat_diff.get(key)
        content_2.append(value - human_ref_value)

accuracies_2 = []
for model_name in model_names:
    accuracies_2.append(second_set.get(model_name))

run_df = pd.DataFrame(
    {
        "accuracy": accuracies,
        "model_name": model_names,
        "model_size": model_sizes,
        "accuracies_2": accuracies_2,
    }
)

run_df_2 = pd.DataFrame(
    {"model_name": model_names_clean, **formatted_accuracies_subcat}
)

run_df_2 = run_df_2.sort_values(
    by=["model_name"],
    ascending=True,
)
print(
    run_df_2.to_latex(
        column_format="lc",
        index=False,
        float_format="%.2f",
    )
)

run_df_3 = pd.DataFrame(
    {"model_name": model_names_clean, **formatted_accuracies_subcat_diff}
)

run_df_3 = run_df_3.sort_values(
    by=["model_name"],
    ascending=True,
)
print(
    run_df_3.to_latex(
        column_format="lc",
        index=False,
        float_format="%.2f",
    )
)


def format_numbers(num: int, max_fraction_digits: int = 1):
    digits = f"{num:,}"
    comma = digits.count(",")
    seperates = {
        "1": "K",
        "2": "M",
        "3": "B",
        "4": "T",
        "5": "Q",
        "6": "S",
        "7": "O",
        "8": "N",
    }
    seperate = seperates.get(str(comma), "N/A")
    if seperate == "N/A":
        return digits
    second = digits.split(",")[1]
    if max_fraction_digits and (p := second[:max_fraction_digits]) != "0":
        return f'{digits[:digits.find(",")]}.{p}{seperate}'
    else:
        return f'{digits[:digits.find(",")]}{seperate}'


llm_stats = run_df[["model_name", "model_size"]]

for idx, row in llm_stats.iterrows():
    model_name_clean = (
        r"\texttt{"
        + row["model_name"]
        .split("/")[-1]
        .capitalize()
        .replace("bert", "BERT")
        .replace("Bert", "BERT")
        .replace("gpt", "GPT")
        .replace("french", "French")
        .replace("-unsloth-bnb-4bit", "")
        .replace("Gpt", "GPT")
        .replace("-bnb-4bit", "")
        .replace("instruct", "it")
        .replace("Xlm", "XLM")
        .replace("Flan-t5", "FLAN-T5")
        .replace("Deepseek", "DeepSeek")
        .replace("llama", "Llama")
        .replace("alpaca", "Alpaca")
        .replace("Qwq", "QwQ")
        .replace("Smollm", "SmolLM")
        + r"}"
    )
    model_weights_clean = format_numbers(row["model_size"])

    llm_stats.loc[idx] = {
        "model_name": model_name_clean,
        "model_size": model_weights_clean,
    }
llm_stats["source"] = ""

llm_stats = llm_stats.sort_values(
    by=["model_name"],
    ascending=True,
)
llm_stats = llm_stats[["model_name", "source", "model_size"]]
print(
    llm_stats.to_latex(
        columns=["model_name", "source", "model_size"],
        column_format="llc",
        index=False,
    )
)
random_accuracy_value = list(run_df["accuracy"][run_df["model_name"] == "Aléatoire"])[0]
if lang == "fr":
    annotators_accuracy_value = list(
        run_df["accuracy"][run_df["model_name"] == "Annotateurs"]
    )[0]

# We remove "aléatoire and annotateurs" since they do not have params value.
run_df = run_df.loc[run_df["model_name"] != "Aléatoire"]
run_df = run_df.loc[run_df["model_name"] != "Annotateurs"]

# We base the color on whether the model is specifically train on the language or not

run_df["color"] = np.where(
    run_df["model_name"] == "Aléatoire",
    "red",
    np.where(run_df["model_name"] == "Annotateurs", "black", "blue"),
)

fig = px.scatter(
    run_df,
    x="model_size",
    y="accuracy",
    color=run_df["color"],
    trendline="ols",
    trendline_options=dict(log_x=True),
    log_x=True,
    hover_name="model_name",
).update_layout(
    xaxis_title="Model size",
    yaxis_title="Accuracy",
    showlegend=False,
)

fig.update_layout(yaxis_range=[45, 95])
print(min(run_df["model_size"]))
print(max(run_df["model_size"]))

fig = fig.add_hline(
    y=random_accuracy_value,
    line_width=10,
    line_dash="dash",
    line_color="red",
    opacity=1,
)
fig = fig.add_hline(
    y=annotators_accuracy_value,
    line_width=2,
    line_dash="dash",
    line_color="green",
    opacity=1,
)
print(random_accuracy_value)
print(annotators_accuracy_value)

# fig.show(renderer="browser")
tikzplotly.save("fig.tex", fig)

fig.write_html(os.path.join("results", f"minimal_pair_analysis_{lang}.html"))
run_df.drop(["color"], axis=1, inplace=True)

run_df.to_csv(os.path.join("results", f"result_{lang}.tsv"), index=False, sep="\t")
run_df[run_df["accuracy"].ge(annotators_accuracy_value)]["model_name"].to_csv(
    os.path.join("results", f"better_than_human_{lang}.tsv"),
    index=False,
    sep="\t",
)

fig = px.scatter(
    run_df,
    x="accuracy",
    y="accuracies_2",
    hover_name="model_name",
).update_layout(
    xaxis_title="MultiBLiMP",
    yaxis_title="QFrBLiMP",
    showlegend=False,
)

fig.update_layout(yaxis_range=[45, 100], xaxis_range=[45, 100])

import plotly.graph_objs as go

fig = fig.add_trace(go.Scatter(x=[45, 100], y=[45, 100], mode="lines"))

# fig.show(renderer="browser")
tikzplotly.save("fig_2.tex", fig)

fig.write_html(os.path.join("results", f"minimal_pair_analysis_multiblimp.html"))
run_df.to_csv(os.path.join("results", f"result_multiblimp.tsv"), index=False, sep="\t")
run_df[run_df["accuracy"].ge(annotators_accuracy_value)]["model_name"].to_csv(
    os.path.join("results", f"better_than_human_multiblimp.tsv"),
    index=False,
    sep="\t",
)
