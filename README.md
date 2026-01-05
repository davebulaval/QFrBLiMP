# QFrBLiMP: A Quebec-French Benchmark of Linguistic Minimal Pairs

This repository contains the dataset and evaluation code for **QFrBLiMP** (Quebec-French Benchmark of Linguistic Minimal Pairs), as introduced in the paper: **"QFrBLiMP: a Quebec-French Benchmark of Linguistic Minimal Pairs"**.

## Abstract

QFrBLiMP is a corpus designed to evaluate Large Language Models' (LLMs) linguistic knowledge of prominent grammatical phenomena in **Quebec-French**. It is the first large-scale minimal pair dataset for this variety of French.

The dataset comprises **1,761 minimal pairs** (MPs) covering **20 linguistic phenomena**. Each pair consists of a grammatical sentence and an ungrammatical counterpart, created by manually modifying sentences extracted from the *Banque de dépannage linguistique* (BDL), an official resource maintained by the *Office québécois de la langue française* (OQLF).

To provide a robust baseline, each pair was annotated by 12 native Quebec-French speakers.

## Dataset Statistics

<img width="254" height="59" alt="image" src="https://github.com/user-attachments/assets/ecb64f60-fd6e-4bd4-9a45-354db7afa7d3" />

Available on [HuggingFace](https://huggingface.co/datasets/davebulaval/qfrblimp).

* **Language:** Quebec-French (Fr-Qc).
* **Total Pairs:** 1,761.
* **Linguistic Phenomena (LP):** 20 categories.
* **Human Annotation:** 12 native speakers per pair.
* **Source:** *Banque de dépannage linguistique* (BDL).

### Linguistic Phenomena Covered

The dataset covers the following 20 linguistic phenomena (LP) :

| ID | Linguistic Phenomena (French) | English Translation | # Pairs |
| --- | --- | --- | --- |
| 1 | Accords participes passés | Past participle agreements | 97 |
| 2 | Flexion du verbe | Verb inflection | 95 |
| 3 | ne ... que | only ... that | 97 |
| 4 | Sélection morphologie fonctionnelle | Functional morphology selection | 96 |
| 5 | Clitique dans la négation de l'infinitif | Clitics in infinitive negation | 99 |
| 6 | Montée du clitique | Rising clitics | 97 |
| 7 | Négation standard | Standard negation | 114 |
| 8 | Déterminants | Determinants | 106 |
| 9 | Sémantique lexicale | Lexical semantics | 113 |
| 10 | Accord dans l'expression idiomatique | Agreement in idiomatic expression | 91 |
| 11 | Accord des adjectifs | Adjective agreement | 100 |
| 12 | -é / -er | -é / -er distinction | 100 |
| 13 | Sélection lexicale du complément | Lexical selection of the complement | 96 |
| 14 | Négation de l'infinitif | Infinitive negation | 97 |
| 15 | Îlot sujet | Subject island | 60 |
| 16 | Îlot ajout | Adjunct island | 60 |
| 17 | Îlot qu- | Wh-island | 60 |
| 18 | Îlot SN | Complex NP island | 60 |
| 19 | Dépendance parasitique avec dont | Parasitic dependence with *dont* | 60 |
| 20 | Préposition orpheline | Orphan preposition | 63 |

## Usage

### Loading the Data

You can load the data using the `datasets` library from Hugging Face (assuming the dataset is hosted there) or directly from this repository.

```python
from datasets import load_dataset

# Load from Hugging Face
dataset = load_dataset("davebulaval/davebulaval/qfrblimp") # Update with actual path

# View an example
print(dataset['train'][0])
# Output:
# {
#   'sentence_good': "La loi fédérale n'inclut pas les lois provinciales.",
#   'sentence_bad': "La loi fédérale, incluant les lois provinciales.",
#   'linguistic_phenomenon': "Accords participes passés",
#   'label': 1
# }

```

### Evaluation Method

Following standard BLiMP methodology, an LLM is considered to have "acquired" a grammatical rule if it assigns a higher probability (lower perplexity) to the grammatical sentence than to the ungrammatical one.

The perplexity (PPL) is calculated as follows:

## Experiments & Results

We benchmarked **77 open-source LLMs** (ranging from <1B to 72B parameters) on QFrBLiMP.

**Key Findings:**

1. **Scaling Laws:** Grammatical competence generally scales with model size.
2. **Difficulty Hierarchy:** Models master frequent morphological rules (e.g., *-é/-er*) but consistently fail on phenomena requiring deep semantic understanding (e.g., Lexical Semantics, Orphan Prepositions).
3. **Dialectal Gap:** Most models show significant performance degradation on Quebec-French compared to Metropolitan French (MultiBLiMP), though the most capable models demonstrate cross-dialectal robustness.

*For detailed results, please refer to Section 5 and Table 11 of the paper.*

## Citation

If you use QFrBLiMP in your research, please cite our paper:

```bibtex
...

```

## License

The QFrBLiMP dataset is released under the **CC BY-NC 4.0** license.
The source sentences were extracted from the *Banque de dépannage linguistique* (BDL) with authorization.
