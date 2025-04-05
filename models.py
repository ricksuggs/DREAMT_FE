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
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm  # Add this import
from utils import *
import math
import warnings
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    def __init__(self, input_size, d_model=512, nhead=8, num_layers=6, dropout=0.2, max_seq_length=2000):
        super().__init__()
        
        # Project input features to transformer dimension
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding for temporal information
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder with increased capacity
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,  # Increased from 2*d_model
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Additional layers for better feature extraction
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model, 2)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters with improved initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode='fan_out')
    
    def forward(self, x, lengths, src_key_padding_mask=None):
        # Project input to transformer dimension
        x = self.input_proj(x)
        x = self.layer_norm1(x)
        x = self.dropout1(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask if not provided
        if src_key_padding_mask is None:
            src_key_padding_mask = create_padding_mask(lengths, x.device)
        
        # Apply transformer with padding mask
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Additional feature processing
        x = self.layer_norm2(x)
        x = self.dropout2(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        
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

def create_padding_mask(lengths, device):
    """Create padding mask for transformer based on sequence lengths"""
    # Convert lengths to tensor and move to correct device if needed
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, device=device)
    else:
        lengths = lengths.to(device)
    
    max_len = int(lengths.max().item())
    mask = torch.arange(max_len, device=device)[None, :] >= lengths[:, None]
    return mask


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


def Transformer_engine(dataloader_train, num_epoch, d_model=128, nhead=4, num_layers=2, learning_rate=0.0001, accumulation_steps=1, dropout=0.1):
    """Train a Transformer model using a DataLoader with gradient accumulation and memory optimization.
    
    Parameters
    ----------
    dataloader_train : DataLoader
        DataLoader for training data
    num_epoch : int
        Number of epochs to train
    d_model : int
        Model dimension, defaults to 128
    nhead : int
        Number of attention heads, defaults to 4
    num_layers : int
        Number of transformer layers, defaults to 2
    learning_rate : float
        Learning rate, defaults to 0.0001
    accumulation_steps : int
        Number of steps for gradient accumulation, defaults to 1

    Returns
    -------
    model : SleepTransformer
        Trained transformer model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on {device}")

    try:
        # Initialize model with improved architecture
        input_size = 4
        model = SleepTransformer(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        # Loss function with class weights
        loss_function = nn.CrossEntropyLoss(
            label_smoothing=0.1,
            ignore_index=-100  # Ignore padding tokens
        )

        # Optimizer with better parameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        # Learning rate scheduler with warmup
        total_steps = len(dataloader_train) * num_epoch
        warmup_steps = total_steps // 10

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        scaler = torch.cuda.amp.GradScaler(enabled=True)

        # Training metrics tracking
        best_accuracy = 0.0
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epoch):
            model.train()
            total_loss = 0
            total_accuracy = 0
            num_batches = 0

            train_iterator = tqdm(dataloader_train, desc=f'Epoch {epoch+1}/{num_epoch}')
            
            for i, batch in enumerate(train_iterator):
                try:
                    # Process batch
                    sample = batch["sample"].to(device, non_blocking=True)
                    length = batch["length"]
                    label = batch["label"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                    # Forward pass with mixed precision
                    with torch.cuda.amp.autocast():
                        y_pred = model(
                            sample, 
                            length,
                            src_key_padding_mask=attention_mask
                        )
                        y_pred = y_pred.view(-1, 2)
                        label = label.view(-1)
                        
                        # Calculate loss only on non-padding tokens
                        valid_mask = (label != -100)
                        loss = loss_function(
                            y_pred[valid_mask],
                            label[valid_mask]
                        ) / accumulation_steps

                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    if (i + 1) % accumulation_steps == 0:
                        # Gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        # Optimizer step
                        scaler.step(optimizer)
                        scaler.update()
                        
                        # Zero gradients
                        optimizer.zero_grad(set_to_none=True)
                        
                        # Scheduler step (after optimizer)
                        scheduler.step()

                    # Update metrics
                    total_loss += loss.item() * accumulation_steps
                    accuracy = calculate_accuracy(y_pred[valid_mask], label[valid_mask]).item()
                    total_accuracy += accuracy
                    num_batches += 1

                    # Update progress bar
                    train_iterator.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'accuracy': f'{accuracy:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                    })

                    # Memory cleanup
                    del sample, label, y_pred, loss
                    torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logging.error(f"GPU OOM in batch {i}. Attempting recovery...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            # Calculate epoch metrics
            avg_loss = total_loss / num_batches
            avg_accuracy = total_accuracy / num_batches

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_accuracy = avg_accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Log epoch metrics
            if (epoch + 1) % 5 == 0:
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
    """Evaluate a Transformer model using a DataLoader.
    
    Parameters
    ----------
    transformer_model : SleepTransformer
        The trained transformer model
    dataloader_test : DataLoader
        DataLoader containing test data
    list_true_stages_test : list
        List of true labels for test data
    test_name : str
        Name of the test for logging
        
    Returns
    -------
    DataFrame
        Results dataframe containing evaluation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_model.eval()
    transformer_model.to(device)

    predicted_probabilities_test = []
    kappa = []
    valid_predictions = []
    valid_labels = []

    with torch.no_grad():
        for batch in dataloader_test:
            # Get data from batch
            sample = batch["sample"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass with attention mask
            outputs = transformer_model(
                sample, 
                length,
                src_key_padding_mask=attention_mask
            )
            
            # Get valid predictions (ignore padding)
            valid_mask = ~attention_mask  # Convert to boolean mask
            batch_size = outputs.size(0)
            
            for b in range(batch_size):
                # Get sequence length for this sample
                seq_len = length[b].item()
                
                # Get valid predictions and labels for this sequence
                valid_pred = outputs[b, :seq_len].cpu().numpy()
                valid_lab = label[b, :seq_len].cpu().numpy()
                
                # Store valid predictions and labels
                valid_predictions.append(valid_pred)
                valid_labels.append(valid_lab)
                
                # Calculate kappa for this sequence
                kappa.append(
                    cohen_kappa_score(
                        valid_lab,
                        valid_pred.argmax(axis=-1)
                    )
                )

    # Concatenate all valid predictions and labels
    array_predict = np.concatenate(valid_predictions)
    array_true = np.concatenate(valid_labels)

    # Calculate metrics using only valid predictions
    transformer_test_results_df = calculate_metrics(
        array_true, 
        array_predict, 
        test_name
    )
    transformer_test_results_df["Cohen's Kappa"] = np.mean(kappa)
    
    # Plot confusion matrix
    plot_cm(array_predict, valid_labels, test_name)

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
