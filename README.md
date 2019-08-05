# IEEE Fraud Detection

Can you detect fraud from customer transactions ?

[Link](https://www.kaggle.com/c/ieee-fraud-detection/overview)

## Prerequisites

- [Anaconda](https://www.anaconda.com/download/) >=5.x

## Set up conda environment

```
conda create -n ieee-fraud-detection python=3.6
conda activate ieee-fraud-detection
pip install -r requirements.txt
```

To move beyond notebook prototyping, all reusable code should go into the `src/` folder package. To use that package inside your project, install the project's module in editable mode, so you can edit files in the `src/` folder and use the modules inside your notebooks :

```
pip install --editable .
```

Then launch jupyter notebook with `jupyter notebook` or lab with `jupyter lab`.

## Run scripts

Run full pipeline : `run_experiment --nrows_train 1000 --name test`

```
(ieee-fraud-detection) λ run_experiment --help
Usage: run_experiment [OPTIONS]

Options:
  --nrows_train INTEGER  Number of training rows
  --name TEXT            Submission name
  --help                 Show this message and exit.
```

## Kaggle API credentials

To use the [Kaggle client library](https://github.com/Kaggle/kaggle-api), sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (`https://www.kaggle.com/<username>/account`) and select 'Create API Token'. This will trigger the download of `kaggle.json`, a file containing your API credentials. Place this file in the location `~/.kaggle/kaggle.json` (on Windows in the location `C:\Users\<Windows-username>\.kaggle\kaggle.json`).

For your security, ensure that other users of your computer do not have read access to your credentials. On Unix-based systems you can do this with the following command:

`chmod 600 ~/.kaggle/kaggle.json`

## Submit to Kaggle

```
kaggle competitions submit -c ieee-fraud-detection -f data/submissions/sample_submission.csv -m "My submission message"
```

# Quick Python lines

In a Python kernel in Jupyter

Rebuild dataset in interim :

```py
from src.dataset.make_dataset import Dataset
ds = Dataset()
ds.load_raw()
ds.save_dataset()

ds.load_raw(nrows_train=1000, load_test=False)
ds.save_dataset(version="_1000", save_test=False)
```

# Project organization

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── submissions    <- All predictions to reuse
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        |
        ├── app.py         <- Main entry point
        │
        ├── dataset        <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py

Project based on the [cookiecutter Kaggle template project](https://github.com/andfanilo/cookiecutter-kaggle). #cookiecutterdatascience
