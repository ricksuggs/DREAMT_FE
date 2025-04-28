# Sleep Tracking with Wearables on Small, Diverse Datasets: A Study of Modeling Choices

This repository is a fork of the source code for the paper _Addressing Wearable Sleep Tracking Inequity: A New Dataset and Novel Methods for a Population with Sleep Disorders_, by Will Ke Wang, et al, published in Proceedings of the Fifth Conference on Health, Inference, and Learning, PMLR 248:380-396, 2024.

Reference:

Wang, W.K., Yang, J., Hershkovich, L., Jeong, H., Chen, B., Singh, K., Roghanizad, A.R., Shandhi, M.M.H., Spector, A.R. &amp; Dunn, J.. (2024). Addressing Wearable Sleep Tracking Inequity: A New Dataset and Novel Methods for a Population with Sleep Disorders. <i>Proceedings of the fifth Conference on Health, Inference, and Learning</i>, in <i>Proceedings of Machine Learning Research</i> 248:380-396 Available from https://proceedings.mlr.press/v248/wang24a.html.

Original code repository for the paper: https://github.com/WillKeWang/DREAMT_FE

The motivation for this repository is a replication of the paper's results, and to experiment with some ablations and extensions for academic purposes:

1. Replace LSTM post-processing with Transformer post-processing
2. Use focal loss in place of cross-entropy loss in LSTM post-processing

## Setup

1. Clone this repository.
2. Create a python virtual environment with python 3.10
```sh
    python -m venv .venv
```
3. Install the dependencies
```sh
    pip install -r requirements.txt
```

## Execution

1. Download the dataset from https://physionet.org/content/dreamt/2.0.0/
2. Update `data_folder` in `feature_engineering.py:1387` to `data_64hz` folder in download
3. Run feature engineering script
```sh
    # Expected execution time: ~ 13 hours
    python feature_engineering.py
``` 
4. Feature dataframes will be regenerated in `dataset_sample/features_df`
5. Run the calculate quality scores script
```sh
    python calculate_quality_score.py
```
6. Run the training and evaulation pipeline without 5-fold cross validation
```sh
    # Expected execution time: ~ 1 hour
    python main.py
```
7. Run the training and evaluation pipeline with 5-fold cross-validation
```sh
    # Expected execution time: ~ 5 hours
    python main_cv.py
```

## Directory Structure

The main components of the project pipeline includes: 
* Perform preprocessing and feature enginering on the data
* Training models for classification

```bash
.
├── calculate_quality_score.py
├── compile_aggregate.py
├── dataset_sample
│   ├── E4_aggregate_subsample
│   │   └── subsampled_SID_whole_df.csv
│   ├── features_df
│   │   └── subsampled_SID_whole_df.csv
│   └── participant_info.csv
├── datasets.py
├── experiments.ipynb
├── feature_engineering.py
├── main.py
├── main_cv.py
├── models.py
├── read_raw_e4.py
├── requirements.frozen.txt
├── requirements.txt
├── results
│   ├── quality_scores_per_subject.csv
└── utils.py

```

## Description
`dataset_sample` is the folder containing a sample data folder for feature engineered data for every participant, a file for participant information, and subsampled raw signal data for each participant.  

`features_df` is the folder contarining the files for feature engineered data.  

`sid_domain_features_df.csv` is the csv file containing features calculated from the raw Empatica E4 data recorded during data collection.  

`participant_info.csv` is the csv file containing the basic information of the participant. 

`quality_score_per_subject.csv`: is a file summarizing the percentage of artifacts of each subject's data calculated from features dataframe `sid_domain_features_df.csv`.   

`read_raw_e4.py` is a module that read raw Empatica E4 data, sleeps stage label, and sleep report to generate a dataframe that aligns the Empatica E4 data with sleep stage and sleep performance, such as Apnea-Hypopnea Index, by time.  

`feature_engineering.py` is a module that read the processed data by `read_raw_e4.py` and perform feature engineering on the data. The result data is stored in `feature_df` in `data`. 

`datasets.py` is a module that read the feature engineered data in `feature_df` and perform data loading, cleaning, and resampling. The processed data is split into train, test, and validation set.  

`models.py` is a module that build, train, and test the model using the train, test, and validation set from `datasets.py`. It will return a result metrics and confusion matrix of the model performance.  

`main.py` is a module that run the entire process of data loading, cleaning, splitting, model building, training, testing and evaluating.  

`utils.py` is a script that contains all the helper functions for data loading, cleaning, splitting, model building, training, testing, and evaluating.

`compile_aggregate.py` is a script that aggregates data from the 5-fold cross-validation run to display in a report.
