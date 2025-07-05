import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import wandb

root = "results"
saving_dir = "figs"
listdir = os.listdir(root)

with open(os.path.join("datastore", "models_size.json"), "r") as file:
    models_size = json.load(file)

# langs = ["nl", "en", "zh", "fr", "ja"]

langs = ["nl", "zh", "fr", "ja"]

api = wandb.Api()

for lang in langs:
    # Project is specified by <entity/project-name>
    runs = api.runs(f"davebulaval/minimal_pair_analysis_{lang}")

    accuracies, model_names, model_sizes = [], [], []
    for run in runs:
        if run.state == "finished":
            accuracies.append(
                [
                    v.get("accuracy")
                    for k, v in run.summary._json_dict.items()
                    if not k.startswith("_")
                ][0]
            )

            model_name = [v for k, v in run.config.items() if k == "model_name"][0]
            model_names.append(model_name)

            model_size = models_size.get(model_name)

            model_sizes.append(model_size)

    run_df = pd.DataFrame(
        {"accuracy": accuracies, "model_name": model_names, "model_size": model_sizes}
    )

    random_accuracy_value = list(
        run_df["accuracy"][run_df["model_name"] == "Aléatoire"]
    )[0]
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
        title=dict(
            text=f"{lang.upper()} minimal pair log-scaled X axis and log-transformed fit",
            font=dict(size=24, color="#000000"),
            y=1,
            x=0.5,
            xanchor="center",
            yanchor="top",
        ),
        xaxis_title="Model size",
        yaxis_title="Accuracy",
        showlegend=False,
    )

    # fig.update_layout(yaxis_range=[25, 90])

    fig.add_hline(
        y=random_accuracy_value, line_width=1, line_dash="dash", line_color="red"
    )
    if lang == "fr":
        fig.add_hline(
            y=annotators_accuracy_value,
            line_width=1,
            line_dash="dash",
            line_color="green",
        )

        print(annotators_accuracy_value)

    fig.show(renderer="browser")

    run_df.drop(["color"], axis=1, inplace=True)

    run_df.to_csv(os.path.join("results", f"result_{lang}.tsv"), index=False, sep="\t")
    if lang == "fr":
        run_df[run_df["accuracy"].ge(annotators_accuracy_value)]["model_name"].to_csv(
            os.path.join("results", f"better_than_human_{lang}.tsv"),
            index=False,
            sep="\t",
        )
    fig.write_html(os.path.join("results", f"minimal_pair_analysis_{lang}.html"))
