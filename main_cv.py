# load packages
import pandas as pd
import numpy as np
import shap
import logging
from sklearn.model_selection import KFold

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

np.random.seed(0)

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
logging.info(SW_df.shape)
logging.info(SW_df.Sleep_Stage.value_counts())

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=0)

group_variables = ["AHI_Severity", "Obesity"]
group_variable = get_variable(group_variables, idx=1)
logging.info(len(final_features))
result_dfs = []

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


logging.info("Performing k-fold cross-validation splitting")
for fold, (trainval_idx, test_idx) in enumerate(kf.split(good_quality_sids)):
    
    logging.info("Splitting data into train, validation, and test sets")
    # Since trainval_idx corresponds to 64 subjects, we'll use the first 48 for training and the next 16 for validation
    train_sids = [good_quality_sids[idx] for idx in trainval_idx[:54]]
    val_sids = [good_quality_sids[idx] for idx in trainval_idx[54:]]
    test_sids = [good_quality_sids[idx] for idx in test_idx]

    logging.info(f"Fold {fold + 1}")

    X_train, y_train, group_train = train_test_split(SW_df, train_sids, final_features, group_variable)
    X_val, y_val, group_val = train_test_split(SW_df, val_sids, final_features, group_variable)
    X_test, y_test, group_test = train_test_split(SW_df, test_sids, final_features, group_variable)

    logging.info("Resampling the training data")
    X_train_resampled, y_train_resampled, group_train_resampled = resample_data(X_train, y_train, group_train, group_variable)

    lgb_test_results_df = None
    lgb_lstm_cross_entropy_test_results_df = None
    lgb_lstm_cross_entropy_opt_test_results_df = None
    lgb_lstm_focal_test_results_df = None
    lgb_transformer_test_results_df = None
    gpb_test_results_df = None
    gpb_lstm_cross_entropy_test_results_df = None
    gpb_lstm_cross_entropy_opt_test_results_df = None
    gpb_lstm_focal_test_results_df = None
    gpb_transformer_test_results_df = None

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

                logging.info("Evaluating LSTM model for LightGBM_CrossEntropy without optimal threshold")
                lgb_lstm_cross_entropy_test_results_df = LSTM_eval(
                    LSTM_model,
                    dataloader_test,
                    true_ls_test,
                    'LightGBM_LSTM_CrossEntropy'
                )

                logging.info("Evaluating LSTM model for LightGBM_CrossEntropy with optimal threshold")
                lgb_lstm_cross_entropy_opt_test_results_df = LSTM_eval(
                    LSTM_model,
                    dataloader_test,
                    true_ls_test,
                    'LightGBM_LSTM_CrossEntropy_Opt',
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
                    optimal_threshold=optimal_threshold_focal
                )

        if 'LightGBM_Transformer' in steps:
            logging.info("Creating Transformer dataset for LightGBM")
            dataloader_train = Transformer_dataloader(prob_ls_train, len_train, true_ls_train, batch_size=16)
            dataloader_val = Transformer_dataloader(prob_ls_val, len_val, true_ls_val, batch_size=1)
            dataloader_test = Transformer_dataloader(prob_ls_test, len_test, true_ls_test, batch_size=1)

            logging.info("Running Transformer model for LightGBM post-processing")
            class_ratios = calculate_class_ratios(true_ls_train)
            transformer_model = Transformer_engine(
                dataloader_train,
                d_model=256, nhead=8, num_layers=4, num_epoch=150,
                class_ratios=class_ratios
            )

            logging.info("Finding optimal threshold for LightGBM_Transformer")
            optimal_threshold_transformer = find_optimal_threshold(
                transformer_model,
                dataloader_val,
                device,
                metric='f1'
            )

            logging.info("Evaluating Transformer model for LightGBM post-processing")
            lgb_transformer_test_results_df = Transformer_eval(
                transformer_model,
                dataloader_test,
                true_ls_test,
                'LightGBM_Transformer',
                optimal_threshold=optimal_threshold_transformer
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

                gpb_lstm_cross_entropy_test_results_df = LSTM_eval(
                    LSTM_model, 
                    dataloader_test, 
                    true_ls_test, 
                    'GPBoost_LSTM_CrossEntropy'
                )

                logging.info("Finding optimal threshold for GPBoost_LSTM_CrossEntropy")
                optimal_threshold_ce = find_optimal_threshold(LSTM_model, dataloader_val, device, metric='f1')

                gpb_lstm_cross_entropy_opt_test_results_df = LSTM_eval(
                    LSTM_model, 
                    dataloader_test, 
                    true_ls_test, 
                    'GPBoost_LSTM_CrossEntropy_Opt',
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
            dataloader_train = Transformer_dataloader(prob_ls_train, len_train, true_ls_train, batch_size=16)
            dataloader_val = Transformer_dataloader(prob_ls_val, len_val, true_ls_val, batch_size=1)
            dataloader_test = Transformer_dataloader(prob_ls_test, len_test, true_ls_test, batch_size=1)

            logging.info("Running Transformer model for GPBoost post-processing")
            class_ratios = calculate_class_ratios(true_ls_train)
            transformer_model = Transformer_engine(
                dataloader_train,
                d_model=256, nhead=8, num_layers=4, num_epoch=150,
                class_ratios=class_ratios
            )

            logging.info("Finding optimal threshold for GPBoost_Transformer")
            optimal_threshold_transformer = find_optimal_threshold(
                transformer_model,
                dataloader_val,
                device,
                metric='f1'
            )

            logging.info("Evaluating Transformer model for GPBoost post-processing")
            gpb_transformer_test_results_df = Transformer_eval(
                transformer_model,
                dataloader_test,
                true_ls_test,
                'GPBoost_Transformer',
                optimal_threshold=optimal_threshold_transformer
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
        result_dfs.append(overall_result)
        overall_result.to_csv(f'./results/fold_{fold + 1}_results_{"__".join(group_variable)}.csv', index=False)
