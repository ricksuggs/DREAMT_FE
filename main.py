
# To run in Google Colab, uncomment the following lines
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/DREAMT_FE-dreamt_time_series_transformer/
# !pip install -r requirements.txt

# load packages
import pandas as pd
import numpy as np
import random
import shap
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Importing modules")

from utils import *
from models import *
from datasets import *

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

steps = [
    'LightGBM', 
    'LightGBM_LSTM_CrossEntropy',
    'LightGBM_LSTM_Focal',
    'LightGBM_Transformer',
    'GPBoost',
    'GPBoost_LSTM_CrossEntropy',
    'GPBoost_LSTM_Focal',
    'GPBoost_Transformer',
]

lgb_test_results_df = None
lgb_lstm_cross_entropy_test_results_df = None
lgb_lstm_focal_test_results_df = None
lgb_transformer_test_results_df = None
gpb_test_results_df = None
gpb_lstm_cross_entropy_test_results_df = None
gpb_lstm_focal_test_results_df = None
gpb_transformer_test_results_df = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prepare the data
# Adjust your path here
quality_df_dir = './results/quality_scores_per_subject.csv'
features_dir = "dataset_sample/features_df/"
info_dir = "dataset_sample/participant_info.csv"

logging.info("Preparing the data")
clean_df, new_features, good_quality_sids = data_preparation(
    threshold = 0.2, 
    quality_df_dir = quality_df_dir,
    features_dir = features_dir,
    info_dir = info_dir)

logging.info("Splitting data into train, validation, and test sets")
SW_df, final_features = split_data(clean_df, good_quality_sids, new_features)

random.seed(0)
train_sids = random.sample(good_quality_sids, 56)
remaining_sids = [subj for subj in good_quality_sids if subj not in train_sids]
val_sids = random.sample(remaining_sids, 8)
test_sids = [subj for subj in remaining_sids if subj not in val_sids]

group_variables = ["AHI_Severity", "Obesity"]
# when idx == 0, it returns ['AHI_Severity'], the first variable in the list
# when idx == 1, it returns ['Obesity'], the second variable in the list
group_variable = get_variable(group_variables, idx=0)

X_train, y_train, group_train = train_test_split(SW_df, train_sids, final_features, group_variable)
X_val, y_val, group_val = train_test_split(SW_df, val_sids, final_features, group_variable)
X_test, y_test, group_test = train_test_split(SW_df, test_sids, final_features, group_variable)

logging.info("Resampling the training data")
X_train_resampled, y_train_resampled, group_train_resampled = resample_data(X_train, y_train, group_train, group_variable)

if 'LightGBM' in steps:
    logging.info("Running LightGBM model")
    final_lgb_model = LightGBM_engine(X_train_resampled, y_train_resampled, X_val, y_val)

    logging.info("Calculating training scores for LightGBM model")
    prob_ls_train, len_train, true_ls_train = compute_probabilities(
        train_sids, SW_df, final_features, "lgb", final_lgb_model, group_variable)
    lgb_train_results_df = LightGBM_result(final_lgb_model, X_train, y_train, prob_ls_train, true_ls_train)

    logging.info("Calculating validation scores for LightGBM model")
    prob_ls_val, len_val, true_ls_val = compute_probabilities(
        val_sids, SW_df, final_features, "lgb", final_lgb_model, group_variable
    )

    logging.info("Calculating testing scores for LightGBM model")
    prob_ls_test, len_test, true_ls_test = compute_probabilities(
        test_sids, SW_df, final_features, "lgb", final_lgb_model, group_variable)
    lgb_test_results_df = LightGBM_result(final_lgb_model, X_test, y_test, prob_ls_test, true_ls_test)

    logging.info("Identifying best features using SHAP")
    explainer = shap.TreeExplainer(final_lgb_model)
    shap_values = explainer.shap_values(X_train)
    # shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=final_features)

    if 'LightGBM_LSTM_CrossEntropy' in steps or 'LightGBM_LSTM_Focal' in steps:
        logging.info("Creating LSTM dataset for LightGBM")
                
        dataloader_train = LSTM_dataloader(prob_ls_train, len_train, true_ls_train, batch_size=32)
        dataloader_val = LSTM_dataloader(prob_ls_val, len_val, true_ls_val, batch_size=1)
        dataloader_test = LSTM_dataloader(prob_ls_test, len_test, true_ls_test, batch_size=1)

        if 'LightGBM_LSTM_CrossEntropy' in steps:

            logging.info("Running LSTM model for LightGBM_CrossEntropy")
            LSTM_model = LSTM_engine(
                dataloader_train,
                num_epoch=300,
                hidden_layer_size=32,
                learning_rate=0.001,
                loss='cross_entropy'
            )

            logging.info("Finding optimal threshold for LightGBM_LSTM_CrossEntropy")
            optimal_threshold_ce = find_optimal_threshold(LSTM_model, dataloader_val, device, metric='f1')

            logging.info("Evaluating LSTM model for LightGBM_CrossEntropy")
            lgb_lstm_cross_entropy_test_results_df = LSTM_eval(
                LSTM_model,
                dataloader_test,
                true_ls_test,
                'LightGBM_LSTM_CrossEntropy',
                optimal_threshold=optimal_threshold_ce 
            )

        if 'LightGBM_LSTM_Focal' in steps:

            class_ratios = calculate_class_ratios(true_ls_train)

            logging.info("Running LSTM model for LightGBM_Focal")
            LSTM_model = LSTM_engine(
                dataloader_train, 
                num_epoch=300, 
                hidden_layer_size=32, 
                learning_rate=0.0005,
                loss='focal',
                class_ratios=class_ratios
            )

            logging.info("Finding optimal threshold for LightGBM_LSTM_Focal")
            optimal_threshold_focal = find_optimal_threshold(LSTM_model, dataloader_val, device, metric='f1')

            logging.info("Evaluating LSTM model for LightGBM_Focal on test set")
            lgb_lstm_focal_test_results_df = LSTM_eval(
                LSTM_model,
                dataloader_test,
                true_ls_test,
                'LightGBM_LSTM_Focal',
                optimal_threshold=optimal_threshold_focal # Pass the found threshold
            )

    if 'LightGBM_Transformer' in steps:
        logging.info("Creating Transformer dataset for LightGBM")
        # Ensure dataloaders use shuffle=False for val/test
        dataloader_train = Transformer_dataloader(prob_ls_train, len_train, true_ls_train, batch_size=16)
        dataloader_val = Transformer_dataloader(prob_ls_val, len_val, true_ls_val, batch_size=1) # Added val dataloader
        dataloader_test = Transformer_dataloader(prob_ls_test, len_test, true_ls_test, batch_size=1)

        logging.info("Running Transformer model for LightGBM post-processing")
        class_ratios = calculate_class_ratios(true_ls_train) # Calculate class ratios if needed
        transformer_model = Transformer_engine(
            dataloader_train,
            d_model=256, nhead=8, num_layers=4, num_epoch=150,
            class_ratios=class_ratios # Assuming Transformer_engine uses FocalLoss
        )

        logging.info("Finding optimal threshold for LightGBM_Transformer")
        # *** Use the correct model (transformer_model) and dataloader (dataloader_val) ***
        optimal_threshold_transformer = find_optimal_threshold(
            transformer_model, # Use the trained transformer model
            dataloader_val,    # Use the validation dataloader
            device,
            metric='f1'
        )

        logging.info("Evaluating Transformer model for LightGBM post-processing")
        lgb_transformer_test_results_df = Transformer_eval(
            transformer_model,
            dataloader_test,
            true_ls_test,
            'LightGBM_Transformer',
            optimal_threshold=optimal_threshold_transformer # Pass the found threshold
        )

if 'GPBoost' in steps:
    logging.info("Running GPBoost model")
    final_gpb_model = GPBoost_engine(X_train_resampled, group_train_resampled, y_train_resampled, X_val, y_val, group_val)

    logging.info("Calculating training scores for GPBoost model")
    prob_ls_train, len_train, true_ls_train = compute_probabilities(
        train_sids, SW_df, final_features, 'gpb', final_gpb_model, group_variable)

    gpb_train_results_df = GPBoost_result(final_gpb_model, X_train, y_train, group_train, prob_ls_train, true_ls_train)

    logging.info("Calculating validation scores for GPBoost model")
    prob_ls_val, len_val, true_ls_val = compute_probabilities(
        val_sids, SW_df, final_features, "gpb", final_gpb_model, group_variable
    )

    logging.info("Calculating testing scores for GPBoost model")
    prob_ls_test, len_test, true_ls_test = compute_probabilities(
        test_sids, SW_df, final_features, 'gpb', final_gpb_model, group_variable)
    gpb_test_results_df = GPBoost_result(final_gpb_model, X_test, y_test, group_test, prob_ls_test, true_ls_test)

    if 'GPBoost_LSTM_CrossEntropy' in steps or 'GPBoost_LSTM_Focal' in steps:

        logging.info("Creating LSTM dataset for GPBoost_LSTM_CrossEntropy and GPBoost_LSTM_Focal")
        dataloader_train = LSTM_dataloader(prob_ls_train, len_train, true_ls_train, batch_size=32)
        dataloader_val = LSTM_dataloader(prob_ls_val, len_val, true_ls_val, batch_size=1)
        dataloader_test = LSTM_dataloader(prob_ls_test, len_test, true_ls_test, batch_size=1)

        if 'GPBoost_LSTM_CrossEntropy' in steps:

            logging.info("Running LSTM model for GPBoost_LSTM_CrossEntropy")
            LSTM_model = LSTM_engine(
                dataloader_train, 
                num_epoch=300, 
                hidden_layer_size=32, 
                learning_rate=0.001,
                loss='cross_entropy'
            )

            logging.info("Finding optimal threshold for GPBoost_LSTM_CrossEntropy")
            optimal_threshold_ce = find_optimal_threshold(LSTM_model, dataloader_val, device, metric='f1')

            gpb_lstm_cross_entropy_test_results_df = LSTM_eval(
                LSTM_model, 
                dataloader_test, 
                true_ls_test, 
                'GPBoost_LSTM_CrossEntropy',
                optimal_threshold=optimal_threshold_ce
            )

        if 'GPBoost_LSTM_Focal' in steps:

            class_ratios = calculate_class_ratios(true_ls_train)
            logging.info("Running LSTM model for GPBoost_LSTM_Focal")
            LSTM_model = LSTM_engine(
                dataloader_train, 
                num_epoch=300, 
                hidden_layer_size=32, 
                learning_rate=0.0005, 
                loss='focal',
                class_ratios=class_ratios
            )

            logging.info("Finding optimal threshold for GPBoost_LSTM_Focal")
            optimal_threshold_focal = find_optimal_threshold(LSTM_model, dataloader_val, device, metric='f1')
            gpb_lstm_focal_test_results_df = LSTM_eval(
                LSTM_model, 
                dataloader_test, 
                true_ls_test, 
                'GPBoost_LSTM_Focal',
                optimal_threshold=optimal_threshold_focal
            )

    if 'GPBoost_Transformer' in steps:
        logging.info("Creating Transformer dataset for GPBoost post-processing")
        # Ensure dataloaders use shuffle=False for val/test
        dataloader_train = Transformer_dataloader(prob_ls_train, len_train, true_ls_train, batch_size=16)
        dataloader_val = Transformer_dataloader(prob_ls_val, len_val, true_ls_val, batch_size=1)
        dataloader_test = Transformer_dataloader(prob_ls_test, len_test, true_ls_test, batch_size=1)

        logging.info("Running Transformer model for GPBoost post-processing")
        class_ratios = calculate_class_ratios(true_ls_train)
        # Pass class_ratios if Transformer_engine uses them (e.g., for FocalLoss)
        transformer_model = Transformer_engine(
            dataloader_train,
            d_model=256, nhead=8, num_layers=4, num_epoch=150,
            class_ratios=class_ratios # Assuming Transformer_engine uses FocalLoss
        )

        logging.info("Finding optimal threshold for GPBoost_Transformer")
        # *** Use the correct model (transformer_model) and dataloader (dataloader_val) ***
        optimal_threshold_transformer = find_optimal_threshold(
            transformer_model, # Use the trained transformer model
            dataloader_val,    # Use the validation dataloader
            device,
            metric='f1'
        )

        logging.info("Evaluating Transformer model for GPBoost post-processing")
        gpb_transformer_test_results_df = Transformer_eval(
            transformer_model,
            dataloader_test,
            true_ls_test,
            'GPBoost_Transformer',
            optimal_threshold=optimal_threshold_transformer # Pass the found threshold
        )

# overall result
overall_result = pd.concat([
    lgb_test_results_df, 
    lgb_lstm_cross_entropy_test_results_df,
    lgb_lstm_focal_test_results_df,
    lgb_transformer_test_results_df,
    gpb_test_results_df,
    gpb_lstm_cross_entropy_test_results_df,
    gpb_lstm_focal_test_results_df,
    gpb_transformer_test_results_df,
])
print(group_variable)
print(overall_result)