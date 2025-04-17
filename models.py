"""
This module provides a set of functions for modeling training and evaluation.

Main Functions:
- transform_data: Converts input data into a specified format.
- validate_data: Checks data against a set of validation rules.
- format_output: Formats data for output based on a specified template.

Usage:
To use these functions, import this script and call the desired function with the appropriate parameters. 

For example:

from model import *

Author: 
License: 
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score
import lightgbm as lgb
import gpboost as gpb
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader
from tqdm import tqdm  # Add this import
from utils import *
import math
import warnings
import logging
from tsai.all import Categorize, TSDatasets, TSClassifier, TSDataLoaders, TSStandardize, TST, Learner, LabelSmoothingCrossEntropyFlat, LabelSmoothingCrossEntropy, RocAucBinary
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.metrics import accuracy
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(1)


class BiLSTMPModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.linear = nn.Linear(
            2 * hidden_layer_size, output_size
        )  # 2 output units for 2 classes

    def forward(self, input_seq, lengths):
        packed_input = pack_padded_sequence(
            input_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.linear(output)  # Shape: [batch_size, seq_len, 2]
        return output


class LSTMPModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_layer_size, batch_first=True, dropout=dropout
        )
        self.linear = nn.Linear(
            hidden_layer_size, output_size
        )  # 2 output units for 2 classes

    def forward(self, input_seq, lengths):
        packed_input = pack_padded_sequence(
            input_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.linear(output)  # Shape: [batch_size, seq_len, 2]
        return output


class SleepTransformer(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=2, dropout=0.5, max_seq_length=2000):  # Increased dropout
        super().__init__()
        
        # Project input features to transformer dimension
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding for temporal information
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2*d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, 2)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_padding_mask(lengths, device):
        """Create padding mask for transformer based on sequence lengths"""
        # Convert lengths to tensor and move to correct device if needed
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor(lengths, device=device)
        else:
            lengths = lengths.to(device)
        
        max_len = int(lengths.max().item())
        mask = torch.arange(max_len, device=device)[None, :] >= lengths[:, None]
        return mask
    
    def forward(self, x, lengths, src_key_padding_mask=None):
        """
        Forward pass of the model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_len, input_size]
        lengths : torch.Tensor
            Lengths of sequences in batch
        src_key_padding_mask : torch.Tensor, optional
            Mask for padding tokens (True for padding positions)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, seq_len, 2]
        """
        # Project input to transformer dimension
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # If no mask provided, create one from lengths
        if src_key_padding_mask is None:
            src_key_padding_mask = self._create_padding_mask(lengths, x.device)
        
        # Apply transformer with padding mask
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Apply final linear layer
        x = self.fc(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]

def LightGBM_engine(X_train_resampled, y_train_resampled, X_val, y_val):
    """Train a LightGBM model using hyperparameter optimization.
    
    Parameters
    ----------
    X_train_resampled : array-like
        Training data.
    y_train_resampled : array-like
        Training labels.
    X_val : array-like
        Validation data for early stopping.
    y_val : array-like
        Validation labels for early stopping.

    Returns
    -------
    final_lgb_model : LightGBM model
    """
    space = {
        "max_depth": hp.quniform("max_depth", 2, 6, 1),
        "reg_alpha": hp.quniform("reg_alpha", 0, 180, 2),
        "reg_lambda": hp.uniform("reg_lambda", 0.2, 5),
        "num_leaves": hp.quniform("num_leaves", 20, 100, 10),
        "n_estimators": hp.quniform("n_estimators", 50, 300, 10),
        "learning_rate": hp.uniform("learning_rate", 0.005, 0.5),
    }

    def objective(space):
        clf = lgb.LGBMClassifier(
            objective="binary",
            #is_unbalance=True,
            scale_pos_weight=1.5,
            max_depth=int(space["max_depth"]),
            reg_alpha=space["reg_alpha"],
            reg_lambda=space["reg_lambda"],
            n_estimators=int(space["n_estimators"]),
            learning_rate=space["learning_rate"],
            num_leaves=int(space["num_leaves"]),
            verbose=-1,
        )

        clf.fit(X_train_resampled, y_train_resampled)

        positive_probabilities = clf.predict_proba(X_val)[:, 1]
        predicted_labels = (positive_probabilities > 0.5).astype(int)

        f1 = f1_score(y_val, predicted_labels)
        return {"loss": -f1, "status": STATUS_OK}

    # Run the hyperparameter search
    # trials = Trials()
    # lgb_best_hyperparams = fmin(
    #     fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials
    # )

    lgb_best_hyperparams = {
        'learning_rate': 0.4979172689350684, 
        'max_depth': 2.0,
        'n_estimators': 290.0,
        'num_leaves': 100.0,
        'reg_alpha': 36.0,
        'reg_lambda': 2.0555917249780014
    }
    print("Best hyperparameters:", lgb_best_hyperparams)

    # Adjust the data types of the best hyperparameters
    lgb_best_hyperparams["max_depth"] = int(lgb_best_hyperparams["max_depth"])
    lgb_best_hyperparams["n_estimators"] = int(lgb_best_hyperparams["n_estimators"])
    lgb_best_hyperparams["num_leaves"] = int(lgb_best_hyperparams["num_leaves"])

    final_lgb_model = lgb.LGBMClassifier(
        **lgb_best_hyperparams, random_state=1, num_iterations=50
    )

    final_lgb_model.fit(X_train_resampled, y_train_resampled)

    return final_lgb_model


def LightGBM_predict(final_lgb_model, X_test, y_test):
    """Predict using a trained LightGBM model and calculate evaluation metrics.
    
    Parameters
    ----------
    final_lgb_model : lgb.LGBMClassifier
        LightGBM model
    X_test : array-like
        Data to predict on.
    y_test : array-like
        True labels.

    Returns
    -------
    results_df : DataFrame
        Dataframe with evaluation metrics.
    """
    pred_probabilities = final_lgb_model.predict_proba(X_test)
    results_df = calculate_metrics(y_test, pred_probabilities, "LightGBM")
    return results_df


def LightGBM_result(final_lgb_model, X_test, y_test, prob_ls_test, true_ls_test):
    """Calculate evaluation metrics and plot confusion matrix for a trained LightGBM model.
    
    Parameters
    ----------
    final_lgb_model : lgb.LGBMClassifier
        LightGBM model
    X : array-like
        Data to predict on.
    y : array-like
        True labels.
    prob_ls : array-like
        Predicted probabilities from LightGBM model without post-processing.
    true_ls : array-like
        True labels.

    Returns
    -------
    results_df : DataFrame
        Dataframe with evaluation metrics.
    """
    kappa = calculate_kappa(prob_ls_test, true_ls_test)
    results_df = LightGBM_predict(final_lgb_model, X_test, y_test)
    results_df["Cohen's Kappa"] = kappa
    plot_cm(prob_ls_test, true_ls_test, "LightGBM")

    return results_df


def GPBoost_engine(
    X_train_resampled, group_train_resampled, y_train_resampled, X_val, y_val, group_val
):

    """Train a GPBoost model using hyperparameter optimization.
    
    Parameters
    ----------
    X_train_resampled : array-like
        Training data.
    group_train_resampled : array-like
        Group data for training.
    y_train_resampled : array-like
        Training labels.
    X_val : array-like
        Validation data for early stopping.
    y_val : array-like
        Validation labels for early stopping.
    group_val : array-like
        Group data for validation.

    Returns
    -------
    final_gpb_model : GPBoost model
        the trained GPBoost model
    """
    space = {
        "max_depth": hp.quniform("max_depth", 3, 6, 1),
        "learning_rate": hp.uniform("learning_rate", 0.005, 0.01),
        "num_leaves": hp.quniform("num_leaves", 20, 200, 20),
        "feature_fraction": hp.uniform("feature_fraction", 0.5, 0.95),
        "lambda_l2": hp.uniform("lambda_l2", 1.0, 10.0),
        "lambda_l1": hp.quniform("lambda_l1", 10, 100, 10),
        "pos_bagging_fraction": hp.uniform("pos_bagging_fraction", 0.8, 0.95),
        "neg_bagging_fraction": hp.uniform("neg_bagging_fraction", 0.6, 0.8),
        "num_boost_round": hp.quniform("num_boost_round", 400, 1000, 100),
    }

    def objective(space):
        params = {
            "objective": "binary",
            "max_depth": int(space["max_depth"]),
            "learning_rate": space["learning_rate"],
            "num_leaves": int(space["num_leaves"]),
            "feature_fraction": space["feature_fraction"],
            "lambda_l2": space["lambda_l2"],
            "lambda_l1": space["lambda_l1"],
            "pos_bagging_fraction": space["pos_bagging_fraction"],
            "neg_bagging_fraction": space["neg_bagging_fraction"],
            "num_boost_round": int(space["num_boost_round"]),
            "verbose": -1,
        }
        num_boost_round = params.pop("num_boost_round")

        gp_model = gpb.GPModel(
            group_data=group_train_resampled, likelihood="bernoulli_probit"
        )

        data_train = gpb.Dataset(data=X_train_resampled, label=y_train_resampled)
        clf = gpb.train(
            params=params,
            train_set=data_train,
            gp_model=gp_model,
            num_boost_round=num_boost_round,
        )

        pred_resp = clf.predict(
            data=X_val, group_data_pred=group_val, predict_var=True, pred_latent=False
        )
        positive_probabilities = pred_resp["response_mean"]
        predicted_labels = (positive_probabilities > 0.5).astype(int)

        f1 = f1_score(y_val, predicted_labels)
        return {"loss": -f1, "status": STATUS_OK}

    # Run the hyperparameter search
    # for AHI and obesity together, it's okay to have number of max evaluations be 10 instead of 50
    # due to the much longer fitting time
    trials = Trials()
    gpb_best_hyperparams = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials
    )
    print("Best hyperparameters:", gpb_best_hyperparams)

    # Adjust the types of the best hyperparameters
    gpb_best_hyperparams["max_depth"] = int(gpb_best_hyperparams["max_depth"])
    gpb_best_hyperparams["num_leaves"] = int(gpb_best_hyperparams["num_leaves"])
    gpb_best_hyperparams["num_boost_round"] = int(
        gpb_best_hyperparams["num_boost_round"]
    )

    # Train the final model
    data_train = gpb.Dataset(X_train_resampled, y_train_resampled)
    data_eval = gpb.Dataset(X_val, y_val)
    gp_model = gpb.GPModel(
        group_data=group_train_resampled, likelihood="bernoulli_probit"
    )
    gp_model.set_prediction_data(group_data_pred=group_val)
    evals_result = {}  # record eval results for plotting
    final_gpb_model = gpb.train(
        params=gpb_best_hyperparams,
        train_set=data_train,
        gp_model=gp_model,
        valid_sets=data_eval,
        early_stopping_rounds=10,
        use_gp_model_for_validation=True,
        evals_result=evals_result,
    )

    return final_gpb_model


def GPBoost_predict(final_gpb_model, X_test, y_test, group_test):
    """ Predict using a trained GPBoost model and calculate evaluation metrics.
    
    Parameters
    ----------
    final_gpb_model : gpb.train
        Trained GPBoost model.
    X_test : array-like
        Data to predict on.
    y_test : array-like
        True labels.  
    group_test : array-like
        Group data for prediction.

    Returns
    -------
    gpb_train_results_df : DataFrame
        Dataframe with evaluation metrics.
    """
    pred_resp = final_gpb_model.predict(
        data=X_test, group_data_pred=group_test, predict_var=True, pred_latent=False
    )
    positive_probabilities = pred_resp["response_mean"]
    negative_probabilities = 1 - positive_probabilities
    predicted_probabilities = np.stack(
        [negative_probabilities, positive_probabilities], axis=1
    )
    gpb_train_results_df = calculate_metrics(y_test, predicted_probabilities, "GPBoost")

    return gpb_train_results_df


def GPBoost_result(final_gpb_model, X_test, y_test, group, prob_ls_test, true_ls_test):
    """Calculate evaluation metrics and plot confusion matrix for a trained GPBoost model.
    
    Parameters
    ----------
    final_gpb_model : gpb.train
        Trained GPBoost model.
    X_test : array-like
        Data to predict on.
    y_test : array-like
        True labels.
    group_test : array-like
        Group data for prediction.
    prob_ls_test : array-like
        Predicted probabilities from GPBoost model without post-processing.
    true_ls_test : array-like
        True labels in time series.

    Returns
    -------
    results_df : DataFrame
        Dataframe with evaluation metrics.
    """
    kappa = calculate_kappa(prob_ls_test, true_ls_test)
    results_df = GPBoost_predict(final_gpb_model, X_test, y_test, group)
    results_df["Cohen's Kappa"] = kappa
    plot_cm(prob_ls_test, true_ls_test, "GPBoost")

    return results_df


def LSTM_dataloader(list_probabilities_subject, lengths, list_true_stages, batch_size=1):
    """Create a DataLoader for a list of each subject's data.
    
    Parameters
    ----------
    list_probabilities_subject : list
        List of predicted probabilities for each subject.
    lengths : list
        List of lengths of each subject's data.
    list_true_stages : list
        List of true labels for each subject.

    Returns
    -------
    dataloader : DataLoader
        DataLoader for the LSTM model.
    """
    dataset = TimeSeriesDataset(list_probabilities_subject, lengths, list_true_stages)

    # DataLoader with the custom collate function for handling padding
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader


def LSTM_engine(dataloader_train, num_epoch, hidden_layer_size=32, learning_rate = 0.001):
    """
    Train a LSTM model using a DataLoader.
    
    Parameters
    ----------
    dataloader_train : DataLoader
        DataLoader for the training data.
    num_epoch : int
        Number of epochs to train the model.

    Returns
    -------
    model : BiLSTMPModel
        Trained LSTM model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    input_size = 4  # Number of features
    output_size = 2

    # dropout must be 0 if using only one layer of LSTM
    model = BiLSTMPModel(input_size, hidden_layer_size, output_size).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # Training loop
    epochs = num_epoch

    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        model.train()  # Set the model to training mode

        for i, batch in enumerate(dataloader_train):
            sample = batch["sample"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)

            if sample.shape[1] == 0:
                print("Empty batch detected, skipping...")
                continue

            optimizer.zero_grad()
            y_pred = model(sample, length)

            # Reshape y_pred and label for CrossEntropyLoss
            # CrossEntropyLoss expects y_pred of shape [N, C], label of shape [N]
            y_pred = y_pred.view(-1, 2)  # Flatten output for CrossEntropyLoss
            label = label.view(-1)  # Flatten label tensor

            loss = loss_function(y_pred, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(y_pred, label).item()

        avg_loss = total_loss / len(dataloader_train)
        avg_accuracy = total_accuracy / len(dataloader_train)

        # Optionally, you can calculate loss and accuracy on a validation set he
        # re
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    return model


def LSTM_eval(lstm_model, dataloader_test, list_true_stages_test, test_name):
    """
    Evaluate a LSTM model using a DataLoader.
    
    Parameters
    ----------
    lstm_model : BiLSTMPModel
        Trained LSTM model.
    dataloader_test : DataLoader
        DataLoader for the test data.
    list_true_stages_test : list
        List of true labels for the test data.

    Returns
    -------
    lstm_test_results_df : DataFrame
        Dataframe with evaluation metrics.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model.eval()  # Set the model to evaluation mode
    lstm_model.to(device)

    predicted_probabilities_test = []
    kappa = []

    with torch.no_grad():  # No need to track the gradients
        for batch in dataloader_test:
            sample = batch["sample"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)

            # Forward pass
            outputs = lstm_model(sample, length)

            predicted_probabilities_test.extend(outputs.cpu().numpy())

            # Calculating Cohen's Kappa Score, ensure labels and predictions are on CPU
            kappa.append(
                cohen_kappa_score(
                    label.cpu().numpy()[0], np.argmax(outputs.cpu().numpy()[0], axis=1)
                )
            )

    array_true = np.concatenate(list_true_stages_test)
    array_predict = np.concatenate(predicted_probabilities_test)

    lstm_test_results_df = calculate_metrics(array_true, array_predict, test_name)
    lstm_test_results_df["Cohen's Kappa"] = np.average(kappa)
    plot_cm(array_predict, list_true_stages_test, test_name)

    return lstm_test_results_df


def Transformer_engine(
    dataloader_train, 
    num_epoch, 
    d_model=128,
    nhead=4,
    num_layers=2,
    learning_rate=0.0001,
    dropout=0.5
):
    """Train a Transformer model using a DataLoader."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on {device}")

    try:
        # Initialize model
        input_size = 4
        model = SleepTransformer(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        # Loss function with label smoothing
        loss_function = nn.CrossEntropyLoss(
            label_smoothing=0.2,
            ignore_index=-100
        )

        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.1
        )

        # Learning rate scheduler - fixed from incorrect import
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=num_epoch,
            steps_per_epoch=len(dataloader_train),
            pct_start=0.2,
            div_factor=25,
            final_div_factor=1000
        )

        # Add early stopping
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epoch):
            model.train()
            total_loss = 0
            total_accuracy = 0
            num_batches = 0

            train_iterator = tqdm(dataloader_train, desc=f'Epoch {epoch+1}/{num_epoch}')
            
            for batch in train_iterator:
                # Process batch
                sample = batch["sample"].to(device)
                length = batch["length"]
                label = batch["label"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Clear gradients
                optimizer.zero_grad(set_to_none=True)

                # Forward pass with stronger regularization
                y_pred = model(sample, length, src_key_padding_mask=attention_mask)
                y_pred = y_pred.view(-1, 2)
                label = label.view(-1)
                
                # Calculate loss only on non-padding tokens
                valid_mask = (label != -100)
                loss = loss_function(y_pred[valid_mask], label[valid_mask])

                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced from 1.0
                
                optimizer.step()
                scheduler.step()

                # Update metrics
                total_loss += loss.item()
                accuracy = calculate_accuracy(y_pred[valid_mask], label[valid_mask]).item()
                total_accuracy += accuracy
                num_batches += 1

                train_iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'accuracy': f'{accuracy:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })

                del sample, label, y_pred, loss
                torch.cuda.empty_cache()

            # Calculate average loss
            avg_loss = total_loss / num_batches
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break

            if (epoch + 1) % 5 == 0:
                avg_accuracy = total_accuracy / num_batches
                logging.info(
                    f"Epoch {epoch+1}/{num_epoch} - "
                    f"Loss: {avg_loss:.4f}, "
                    f"Accuracy: {avg_accuracy:.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )

    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        torch.cuda.empty_cache()
        raise e

    return model

def Transformer_eval(transformer_model, dataloader_test, list_true_stages_test, test_name):
    """Evaluate a Transformer model using a DataLoader."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_model.eval()
    transformer_model.to(device)

    predicted_probabilities_test = []
    valid_true_labels = []
    kappa = []
    
    seq_idx = 0
    
    with torch.no_grad():
        for batch in dataloader_test:
            # Get data from batch
            sample = batch["sample"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lengths = batch["length"]
            
            # Forward pass with attention mask
            outputs = transformer_model(
                sample, 
                lengths,
                src_key_padding_mask=attention_mask
            )
            
            batch_size = outputs.size(0)
            for b in range(batch_size):
                # Get sequence length for this sample
                seq_len = lengths[b].item()
                true_labels = list_true_stages_test[seq_idx][:seq_len]  # Truncate to sequence length
                
                # Use attention mask to get valid predictions
                valid_positions = ~attention_mask[b, :seq_len]  # Only use positions up to sequence length
                valid_pred = outputs[b, :seq_len][valid_positions].cpu().numpy()
                
                # Ensure predictions and true labels have same length
                min_len = min(len(true_labels), len(valid_pred))
                true_labels = true_labels[:min_len]
                valid_pred = valid_pred[:min_len]
                
                # Store predictions and labels
                predicted_probabilities_test.append(valid_pred)
                valid_true_labels.append(true_labels)
                
                # Calculate kappa using matched lengths
                pred_labels = valid_pred.argmax(axis=-1)
                kappa.append(cohen_kappa_score(true_labels, pred_labels))
                
                seq_idx += 1

    # Concatenate predictions and true labels
    array_predict = np.concatenate([p.argmax(axis=-1) for p in predicted_probabilities_test])
    array_true = np.concatenate(valid_true_labels)
    array_probabilities = np.concatenate(predicted_probabilities_test)

    # Calculate metrics using both predictions and probabilities
    transformer_test_results_df = calculate_metrics(
        array_true, 
        array_probabilities,  # Use probabilities for metrics that need them
        test_name
    )
    transformer_test_results_df["Cohen's Kappa"] = np.mean(kappa)
    
    # Plot confusion matrix using predictions
    plot_cm(array_predict, array_true, test_name)
    
    return transformer_test_results_df

def Transformer_dataloader(list_probabilities_subject, lengths, list_true_stages, batch_size=16):
    """Create a DataLoader for Transformer models.
    
    Parameters
    ----------
    list_probabilities_subject : list
        List of predicted probabilities for each subject
    lengths : list
        List of sequence lengths
    list_true_stages : list
        List of true labels
    batch_size : int, optional
        Batch size for training, defaults to 16
        
    Returns
    -------
    DataLoader
        DataLoader configured for Transformer input
    """
    class TransformerDataset(Dataset):
        def __init__(self, probabilities, lengths, labels):
            self.probabilities = probabilities
            self.lengths = lengths
            self.labels = labels
            self.max_len = max(lengths)

        def __len__(self):
            return len(self.probabilities)

        def __getitem__(self, idx):
            prob = self.probabilities[idx]
            length = self.lengths[idx]
            label = self.labels[idx]

            # Pad sequences
            if prob.shape[0] < self.max_len:
                pad_length = self.max_len - prob.shape[0]
                prob_pad = np.pad(
                    prob, 
                    ((0, pad_length), (0, 0)), 
                    mode='constant'
                )
                label_pad = np.pad(
                    label, 
                    (0, pad_length), 
                    mode='constant', 
                    constant_values=-100
                )
            else:
                prob_pad = prob
                label_pad = label

            # Create attention mask (False for real tokens, True for padding)
            attention_mask = torch.arange(self.max_len) >= length

            return {
                'sample': torch.FloatTensor(prob_pad),
                'length': length,
                'label': torch.LongTensor(label_pad),
                'attention_mask': attention_mask
            }

    def collate_fn(batch):
        """Custom collate function for transformer batches."""
        batch_size = len(batch)
        max_len = max(item['length'] for item in batch)
        
        # Prepare tensors
        samples = torch.stack([item['sample'] for item in batch])
        lengths = torch.tensor([item['length'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        attention_masks = torch.stack([item['attention_mask'] for item in batch])
        
        return {
            'sample': samples,
            'length': lengths,
            'label': labels,
            'attention_mask': attention_masks
        }

    # Create dataset and dataloader
    dataset = TransformerDataset(
        list_probabilities_subject,
        lengths,
        list_true_stages
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=2
    )

class SleepTSDatasets:
    """Time series datasets for sleep stage classification."""
    
    def __init__(self, 
                probabilities: List[np.ndarray], 
                lengths: List[int], 
                labels: List[np.ndarray],
                max_len: Optional[int] = None):
        """Initialize datasets with proper sequence handling.
        
        Args:
            probabilities: List of arrays with shape (n_timesteps, n_features)
            lengths: List of sequence lengths
            labels: List of label arrays with shape (n_timesteps,)
            max_len: Optional maximum sequence length for padding
        """
        logging.info("" + "="*50)
        logging.info("Initializing SleepTSDatasets...")
        logging.debug("Input validation and shape check:")
        logging.debug(f"Number of sequences: {len(probabilities)}")
        logging.debug(f"Number of lengths: {len(lengths)}")
        logging.debug(f"Number of label sequences: {len(labels)}")
        
        # Verify input shapes match
        assert len(probabilities) == len(lengths) == len(labels), \
            f"Mismatched input lengths: probs={len(probabilities)}, lengths={len(lengths)}, labels={len(labels)}"
        
        # Get sequence information
        logging.debug("Sequence length analysis:")
        logging.debug(f"Min length: {min(lengths)}")
        logging.debug(f"Max length: {max(lengths)}")
        logging.debug(f"Mean length: {np.mean(lengths):.2f}")
        
        # Set maximum sequence length
        self.max_len = max_len if max_len is not None else max(lengths)
        logging.debug(f"Using max_len: {self.max_len}")
        
        # Get dimensions
        n_samples = len(probabilities)
        n_vars = probabilities[0].shape[1]
        logging.debug(f"Input dimensions:")
        logging.debug(f"Number of samples: {n_samples}")
        logging.debug(f"Number of variables: {n_vars}")
        logging.debug(f"First sequence shape: {probabilities[0].shape}")
        logging.debug(f"First label shape: {labels[0].shape}")
        
        # Initialize arrays with proper shapes
        logging.debug("Initializing arrays...")
        X = np.zeros((n_samples, n_vars, self.max_len))
        y = np.zeros((n_samples, self.max_len))
        
        # Process each sequence
        logging.debug("Processing sequences:")
        for i, (prob, length, label) in enumerate(zip(probabilities, lengths, labels)):
            logging.debug(f"Sequence {i}:")
            logging.debug(f"  Input shape: {prob.shape}")
            logging.debug(f"  Length: {length}")
            logging.debug(f"  Label shape: {label.shape}")
            
            # Verify sequence lengths match
            assert prob.shape[0] == len(label), \
                f"Sequence {i}: Mismatched lengths - prob={prob.shape[0]}, label={len(label)}"
            
            # Get current sequence length
            curr_len = min(length, self.max_len)
            logging.debug(f"  Using length: {curr_len}")
            
            # Store features and labels with padding
            X[i, :, :curr_len] = prob[:curr_len].T
            y[i, :curr_len] = label[:curr_len]
            
            # Verify data types
            logging.debug(f"  X dtype: {X.dtype}")
            logging.debug(f"  y dtype: {y.dtype}")
            
            # Log label distribution
            label_counts = np.bincount(label[:curr_len].astype(np.int64))
            logging.debug(f"  Label distribution: {dict(zip(range(len(label_counts)), label_counts))}")
            
            # Progress indicator for large datasets
            if i % 10 == 0:
                logging.debug(f"Processed {i}/{n_samples} sequences")
        
        # Verify final shapes
        logging.debug("Final array shapes:")
        logging.debug(f"X shape: {X.shape}")
        logging.debug(f"y shape: {y.shape}")
        
        # Verify no NaN values
        logging.debug("Checking for NaN values:")
        logging.debug(f"X NaN count: {np.isnan(X).sum()}")
        logging.debug(f"y NaN count: {np.isnan(y).sum()}")
        
        # Create TSDatasets
        logging.debug("Creating TSDatasets...")
        try:
            tfms = [None, None]  # No transforms for now
            self.dsets = TSDatasets(X, y, tfms=tfms)
            logging.debug("TSDatasets creation successful")
            
            # Log dataset properties
            logging.debug("TSDatasets properties:")
            logging.debug(f"Input range: [{X.min():.3f}, {X.max():.3f}]")
            logging.debug(f"Unique labels: {np.unique(y)}")
            
        except Exception as e:
            logging.error(f"Failed to create TSDatasets: {str(e)}")
            raise
        
        logging.debug("="*50 + "")

class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self, class_ratio: float):
        super().__init__()
        
        # Adjust weights to more strongly penalize false positives
        neg_weight = 2.0 / (1.0 - class_ratio)  # Increased weight for negative class
        pos_weight = 1.0 / class_ratio
        
        # Normalize weights
        total = neg_weight + pos_weight
        neg_weight = neg_weight / total
        pos_weight = pos_weight / total
        
        self.class_weights = torch.tensor([neg_weight, pos_weight], dtype=torch.float32)
        
    def forward(self, pred, target, lengths=None):
        pred = pred.float()
        target = target.long()
        
        # Move weights to device
        self.class_weights = self.class_weights.to(pred.device)
        
        # Create mask for valid timesteps
        if lengths is not None:
            mask = torch.arange(pred.size(1))[None, :] < lengths[:, None]
            mask = mask.to(pred.device)
        else:
            mask = torch.ones_like(target, dtype=torch.bool)
            
        # Apply focal loss modification
        ce_loss = F.cross_entropy(
            pred[mask].view(-1, pred.size(-1)),
            target[mask].view(-1),
            weight=self.class_weights,
            reduction='none'
        )
        
        # Get probabilities for focal loss
        pt = torch.exp(-ce_loss)
        # Apply focal loss formula with gamma=2
        focal_loss = ((1 - pt) ** 2) * ce_loss
        
        return focal_loss.mean()
    
class TimeDistributedHead(nn.Module):
    def __init__(self, d_model, num_classes, fc_dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Create layers for time-distributed processing
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x, lengths=None):
        # Input shape: [batch_size, d_model, seq_len]
        x = x.transpose(1, 2)  # [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape
        
        # Create attention mask for padded values
        if lengths is not None:
            mask = torch.arange(seq_len)[None, :] < lengths[:, None]
            mask = mask.to(x.device)
        else:
            mask = None
        
        # Process each timestep
        outputs = []
        for t in range(seq_len):
            curr_x = x[:, t, :]  # [batch_size, d_model]
            
            # Apply mask if available
            if mask is not None:
                curr_mask = mask[:, t]
                curr_x = curr_x * curr_mask.float().unsqueeze(1)
            
            curr_out = self.layers(curr_x)
            outputs.append(curr_out)
        
        return torch.stack(outputs, dim=1)  # [batch_size, seq_len, num_classes]

def TST_learner(
    train_probabilities: List[np.ndarray],
    train_lengths: List[int],
    train_labels: List[np.ndarray],
    val_probabilities: List[np.ndarray],
    val_lengths: List[int],
    val_labels: List[np.ndarray],
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    dropout: float = 0.2,
    fc_dropout: float = 0.3,
    batch_size: int = 16,
    num_epochs: int = 150,
    learning_rate: float = 1e-4
) -> Learner:
    """Create and train a TST model using tsai's implementation."""
    try:
        logging.debug("Initializing TST learner with parameters:")
        logging.debug(f"d_model: {d_model}, nhead: {nhead}, num_layers: {num_layers}")
        logging.debug(f"dropout: {dropout}, fc_dropout: {fc_dropout}")
        logging.debug(f"batch_size: {batch_size}, num_epochs: {num_epochs}")
        
        # Log input shapes and types
        logging.debug("Input data shapes:")
        logging.debug(f"Train probabilities: {len(train_probabilities)} sequences")
        logging.debug(f"Train lengths: {len(train_lengths)} values")
        logging.debug(f"Train labels: {len(train_labels)} sequences")
        logging.debug(f"First train sequence shape: {train_probabilities[0].shape}")
        logging.debug(f"First train label shape: {train_labels[0].shape}")
        
        # Create datasets and dataloaders
        max_len = max(max(train_lengths), max(val_lengths))
        train_datasets = SleepTSDatasets(
            train_probabilities, 
            train_lengths, 
            train_labels,
            max_len=max_len
        )
        val_datasets = SleepTSDatasets(
            val_probabilities, 
            val_lengths, 
            val_labels,
            max_len=max_len
        )
        
        # Create dataloaders with proper transforms
        dls = TSDataLoaders.from_dsets(
            train_datasets.dsets,
            val_datasets.dsets,
            bs=batch_size,
            batch_tfms=TSStandardize(by_var=True)
        )
        
        # Verify dataloader shapes
        batch = next(iter(dls.train))
        x, y = batch
        logging.debug(f"First batch shapes:")
        logging.debug(f"Input (x) shape: {x.shape}")
        logging.debug(f"Label (y) shape: {y.shape}")

        # Calculate class ratio from training labels
        all_labels = np.concatenate([label for label in train_labels])
        class_ratio = np.mean(all_labels)  # Ratio of positive class
        logging.debug(f"Class ratio (positive class): {class_ratio:.3f}")
        
        # Create TST model with correct parameters
        model = TST(
            c_in=dls.vars,          # Number of input features
            c_out=2,                # Binary classification
            seq_len=dls.len,        # Sequence length
            max_seq_len=max_len,    # Maximum sequence length
            d_model=d_model,        # Model dimension (128-1024)
            n_heads=nhead,          # Number of attention heads (8-16)
            d_k=None,              # Let it default to d_model/n_heads
            d_v=None,              # Let it default to d_model/n_heads
            d_ff=d_model*4,              # Feedforward dimension (256-4096)
            dropout=dropout,        # Encoder dropout (0.0-0.3)
            act='gelu',      # Default activation
            n_layers=num_layers,    # Number of encoder layers (2-8)
            fc_dropout=fc_dropout,  # FC layer dropout (0.0-0.8)
            y_range=None           # Not needed for classification
        )

        # Replace the model's head
        model.head = TimeDistributedHead(
            d_model=d_model, 
            num_classes=2, 
            fc_dropout=fc_dropout
        )
        
        # Create learner with sequence loss
        learn = Learner(
            dls, 
            model,
            loss_func=SequenceCrossEntropyLoss(class_ratio=class_ratio),
            metrics=[accuracy],
            cbs=[
                EarlyStoppingCallback(
                    monitor='valid_loss',
                    min_delta=0.001,
                    patience=20
                )
            ]
        )
        
        # Train with one-cycle policy
        logging.debug("Starting training...")
        learn.fit_one_cycle(
            num_epochs,
            learning_rate,
            wd=0.1,
            pct_start=0.3
        )
        
        return learn
    
    except Exception as e:
        logging.error(f"TST training failed with error: {str(e)}")
        logging.error("Error details:", exc_info=True)
        raise e

def TST_eval(
    learner: Learner,
    test_probabilities: List[np.ndarray],
    test_lengths: List[int],
    true_labels: List[np.ndarray],
    test_name: str
) -> pd.DataFrame:
    """
    Evaluate TST model performance with time-distributed predictions.
    """
    logging.debug("Evaluating TST model performance...")
    predictions = []
    kappa_scores = []
    processed_true_labels = []
    
    # Create test dataset with same preprocessing as training
    test_datasets = SleepTSDatasets(
        test_probabilities,
        test_lengths,
        true_labels,
        max_len=learner.dls.len
    )
    
    # Create test dataloader using same transforms as validation set
    bs = learner.dls.valid.bs  # Get batch size from validation dataloader
    test_dl = learner.dls.valid.new(test_datasets.dsets[0])
    
    # Get compute device
    device = next(learner.model.parameters()).device
    
    # Get predictions batch by batch
    learner.model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dl):
            # Move batch to device
            x = batch[0].to(device)
            
            # Get model predictions
            outputs = learner.model(x)  # Shape: [batch_size, seq_len, num_classes]
            
            # Convert to probabilities
            batch_probs = torch.softmax(outputs, dim=-1)
            
            # Process each sequence in the batch
            for seq_idx in range(len(batch_probs)):
                batch_offset = batch_idx * bs  # Use stored batch size
                if batch_offset + seq_idx >= len(test_lengths):
                    break
                    
                # Get current sequence length
                curr_len = test_lengths[batch_offset + seq_idx]
                curr_len = min(curr_len, learner.dls.len)
                
                # Get predictions and true labels for current sequence
                seq_probs = batch_probs[seq_idx, :curr_len].cpu().numpy()
                true_seq = true_labels[batch_offset + seq_idx][:curr_len]
                
                # Calculate kappa score
                pred_labels = seq_probs.argmax(axis=-1)
                kappa = cohen_kappa_score(true_seq, pred_labels)
                kappa_scores.append(kappa)
                
                # Store predictions and true labels
                predictions.append(seq_probs)
                processed_true_labels.append(true_seq)
                
                if seq_idx % 10 == 0:
                    logging.debug(f"Processed sequence {batch_offset + seq_idx}")
                    logging.debug(f"Sequence length: {curr_len}")
                    logging.debug(f"Predictions shape: {seq_probs.shape}")
    
    # Concatenate all predictions and true labels
    array_probabilities = np.concatenate(predictions)
    array_true = np.concatenate(processed_true_labels)
    array_predict = array_probabilities.argmax(axis=-1)
    
    # Log shapes and sample values
    logging.debug("Final evaluation arrays:")
    logging.debug(f"Predictions shape: {array_predict.shape}")
    logging.debug(f"True labels shape: {array_true.shape}")
    logging.debug(f"Probabilities shape: {array_probabilities.shape}")
    
    # Calculate and return metrics
    results_df = calculate_metrics(array_true, array_probabilities, test_name)
    results_df["Cohen's Kappa"] = np.mean(kappa_scores)
    
    # Plot confusion matrix
    plot_cm(array_predict, array_true, test_name)
    
    return results_df