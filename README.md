## Ubaar Competition
Example Data Science flow for Ubaar competition


## Data 
The dataset comes from [Kaggle Ubaar Competition](https://www.kaggle.com/c/ubaar-competition)

You should download it to the directory `dara/raw/` so that the directories 
structure is `dara/raw/ubaar-competition/*.csv`

## Installation

`pip install -r requirements.txt`

Tested using Python 3.7

## Running

### Process
The data science process is performed in the following steps:

- `notebooks/initial_eda.ipynb` - variables analysis
- `notebooks/initial_training.ipynb` - features impact on modeling and possible outcome
- `feature_extraction/features_extraction.py` - versioned scalable features extraction script
- `training.ipynb` - modeling in order to achieve best results

Modeling results are stored at [link](http://ubaar-competition-mlflow-284138417.eu-central-1.elb.amazonaws.com/)

### Additional results

Additionally, there are 3 scripts in `helper_scripts/` directory. They visualize a few aspects in
transports localisations:
- `helper_scripts/compare_train_test_localisations.py` - assure all localisations are covered 
between the train and test sets (generalization on new transport sources)
- `helper_scripts/prices_vis.py` - average price per city visualized on a map
- `helper_scripts/test_clustering.py` - script for clustering analysis


## Results

Results are presented in presentation `results.pdf`. Additionally to the model performance report,
a few conclusions are drawn.


## Server API

