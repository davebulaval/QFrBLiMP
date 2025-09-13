# How to annotate the dataset using Prodigy

Here is the command to self-host Prodigy

```shell
python -m prodigy fr_blimp frblimp_annotations ../datastore/QFrBLiMP/unannotated/fr_blimp_50_sentences.jsonl -F recipe.py 
```

Output the annotation:

```shell
prodigy db-out frblimp_annotations > frblimp_annotations.jsonl
```

## Handling multiple annotators

To handle multiple annotators, we chose to use session rather than multiple process as
suggested [here](https://support.prodi.gy/t/help-multiple-annotators-setup/3343/3).
Thus, for an annotator called dave, the url will look like `http://<ip>>:8080/?session=dave`.

- [https://prodigy.ai/docs/task-routing#work-stealing](https://prodigy.ai/docs/task-routing#work-stealing)
