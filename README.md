# MAD Speech

MAD Speech is compatible with Python 3.10, we did not test any other python versions.

[Read the paper (acl anthology)](https://aclanthology.org/2025.naacl-long.11.pdf)

<p align="justify"> Generative spoken language models produce speech in a wide range of voices, prosody, and recording conditions, seemingly approaching the diversity of natural speech. However, the extent to which generated speech is acoustically diverse remains unclear due to a lack of appropriate metrics. We address this gap by developing lightweight metrics of acoustic diversity, which we collectively refer to as MAD Speech. We focus on measuring five facets of acoustic diversity: voice, gender, emotion, accent, and background noise. We construct the metrics as a composition of specialized, per-facet embedding models and an aggregation function that measures diversity within the embedding space. Next, we build a series of datasets with a priori known diversity preferences for each facet. Using these datasets, we demonstrate that our proposed metrics achieve a stronger agreement with the ground-truth diversity than baselines. Finally, we showcase the applicability of our proposed metrics across several real-life evaluation scenarios. </p>

## Usage

MAD Speech outputs a diversity score for each facet of acoustic diversity given a set of audio samples. These samples are expected to be in *wav* format and in the same directory. Below is the usage of the package:

```
python diversity_scores.py --path_audio /PATH/TO/AUDIO/FOLDER/
```

## Reference

```
@inproceedings{futeral-etal-2025-mad,
    title = "{MAD} Speech: Measures of Acoustic Diversity of Speech",
    author = "Futeral, Matthieu  and
      Agostinelli, Andrea  and
      Tagliasacchi, Marco  and
      Zeghidour, Neil  and
      Kharitonov, Eugene",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.11/",
    doi = "10.18653/v1/2025.naacl-long.11",
    pages = "222--235",
    ISBN = "979-8-89176-189-6",
    abstract = "Generative spoken language models produce speech in a wide range of voices, prosody, and recording conditions, seemingly approaching the diversity of natural speech. However, the extent to which generated speech is acoustically diverse remains unclear due to a lack of appropriate metrics. We address this gap by developing lightweight metrics of acoustic diversity, which we collectively refer to as MAD Speech. We focus on measuring five facets of acoustic diversity: voice, gender, emotion, accent, and background noise. We construct the metrics as a composition of specialized, per-facet embedding models and an aggregation function that measures diversity within the embedding space. Next, we build a series of datasets with a priori known diversity preferences for each facet. Using these datasets, we demonstrate that our proposed metrics achieve a stronger agreement with the ground-truth diversity than baselines. Finally, we showcase the applicability of our proposed metrics across several real-life evaluation scenarios. MAD Speech is made publicly available."
}
```
