# TODO: complete this file.
"""
ensemble.py

Bagging ensemble for MLA Final Project 

- Trains 3 base models on 3 different bootstrap samples from the training set:
    1) user-based KNN (uses sklearn KNNImputer implemented in knn.py)
    2) Item Response Theory (IRT) (item_response.py)
    3) Neural Network AutoEncoder (neural_network.py)

- For each validation/test entry, obtains 3 probability predictions (one per model)
  and averages them to get the ensemble probability. Threshold at 0.5 for final class.

- Prints per-model validation/test accuracy and final ensemble accuracy.
- Prints training logs for IRT (iterations) and NN (epochs).
"""

import os
import numpy as np
import random
import torch

# (full run)
DEMO = False
IRT_ITERATIONS_DEMO = 300    
NN_EPOCHS_DEMO = 50         

# Import utilities and model modules (assumes same folder)
from utils import load_train_csv, load_valid_csv, load_public_test_csv, load_train_sparse
import knn
import item_response
import neural_network


def build_matrix_from_interactions(interactions, num_users, num_questions):
    """
    Build a matrix (users x questions) with NaN for missing entries, and 0/1 for observed.
    interactions is a dict with keys 'user_id','question_id','is_correct'.
    """
    M = np.empty((num_users, num_questions))
    M[:] = np.nan
    for u, q, c in zip(interactions['user_id'], interactions['question_id'], interactions['is_correct']):
        M[u, q] = c
    return M


def bootstrap_interaction_sample(train_data):
    """
    Given train_data dict {user_id, question_id, is_correct}, sample with replacement
    at the interaction level and return a new dict of same structure.
    """
    n = len(train_data['user_id'])
    idx = np.random.choice(np.arange(n), size=n, replace=True)
    boot = {
        'user_id': [train_data['user_id'][i] for i in idx],
        'question_id': [train_data['question_id'][i] for i in idx],
        'is_correct': [train_data['is_correct'][i] for i in idx]
    }
    return boot


def preds_from_knn_matrix(matrix, data_entries, k=11):
    """
    Use the KNN imputer approach to produce probability predictions for data_entries.
    - matrix: numpy array users x questions with NaNs for missing
    - data_entries: dict with lists user_id, question_id, is_correct
    Returns: numpy array of predicted probabilities (len = len(data_entries['user_id']))
    """
    # We reuse knn.KNNImputer style inside knn.user_knn_predict_hanu, but that function also returns accuracy
    # To directly extract predictions, call user_knn_predict_hanu(..., return_confusion=True) which returns preds.
    try:
        acc, cm, preds = knn.user_knn_predict_hanu(matrix, data_entries, k, return_confusion=True)
        # preds returned correspond to data_entries (probabilities)
        return np.array(preds)
    except TypeError:
        # Some implementations may return (acc, cm, preds) or just acc; fallback manual:
        imputer = knn.KNNImputer(n_neighbors=k, metric='nan_euclidean') if hasattr(knn, 'KNNImputer') else None
        # fallback: try direct imputation using sklearn (if available)
        from sklearn.impute import KNNImputer as SKKNN
        imp = SKKNN(n_neighbors=k, metric='nan_euclidean')
        imputed = imp.fit_transform(matrix)
        preds = np.array([imputed[u, q] for u, q in zip(data_entries['user_id'], data_entries['question_id'])])
        return preds


def preds_from_irt(theta, beta, data_entries):
    """Return probability vector from IRT parameters for data_entries."""
    sigmoid = item_response.sigmoid
    preds = [sigmoid(theta[u] - beta[q]) for u, q in zip(data_entries['user_id'], data_entries['question_id'])]
    return np.array(preds)


def preds_from_nn(best_model, zero_train_tensor, data_entries):
    """Return probability vector from best_model (PyTorch) for data_entries.
       zero_train_tensor: torch.FloatTensor, zero-imputed training inputs (users x questions)
    """
    best_model.eval()
    preds = []
    with torch.no_grad():
        for u, q in zip(data_entries['user_id'], data_entries['question_id']):
            inp = zero_train_tensor[u].unsqueeze(0)  # (1 x num_questions)
            out = best_model(inp)  # shape (1 x num_questions)
            prob = out[0, q].item()
            preds.append(prob)
    return np.array(preds)


def evaluate_acc_from_probs(probs, truths):
    preds_bin = probs >= 0.5
    return np.mean(preds_bin == np.array(truths))


def main():
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    # -------------------------
    # Load original data
    # -------------------------
    DATA_PATH = "./data"
    print("Loading data from:", DATA_PATH)
    train_data = load_train_csv(DATA_PATH)
    val_data = load_valid_csv(DATA_PATH)
    test_data = load_public_test_csv(DATA_PATH)
    sparse_full = load_train_sparse(DATA_PATH).toarray()  # full (may contain NaN)
    num_users, num_questions = sparse_full.shape
    print(f"Loaded train interactions: {len(train_data['user_id'])}, users={num_users}, questions={num_questions}")

    # -------------------------
    # Create 3 bootstrap samples (interaction-level bootstrapping)
    # -------------------------
    print("\nCreating 3 bootstrap interaction samples ...")
    boot_datas = [bootstrap_interaction_sample(train_data) for _ in range(3)]

    # For KNN: build matrices from bootstrap interactions
    boot_matrices = [build_matrix_from_interactions(bd, num_users, num_questions) for bd in boot_datas]

    # -------------------------
    # Base model 1: KNN (user-based)
    # -------------------------
    KNN_K = 11  # you can change
    print("\n[BASE 1] KNN (user-based) on bootstrap sample #1")
    # Predictions on validation and test using Imputer fitted on bootstrap matrix
    preds_val_knn = preds_from_knn_matrix(boot_matrices[0], val_data, k=KNN_K)
    preds_test_knn = preds_from_knn_matrix(boot_matrices[0], test_data, k=KNN_K)
    val_acc_knn = evaluate_acc_from_probs(preds_val_knn, val_data['is_correct'])
    test_acc_knn = evaluate_acc_from_probs(preds_test_knn, test_data['is_correct'])
    print(f"KNN (k={KNN_K}) - Val acc: {val_acc_knn:.4f} | Test acc: {test_acc_knn:.4f}")

    # -------------------------
    # Base model 2: IRT
    # -------------------------
    print("\n[BASE 2] IRT (train on bootstrap sample #2)")
    if DEMO:
        irt_iters = IRT_ITERATIONS_DEMO
    else:
        irt_iters = 300
    lr = 0.01
    theta, beta, val_acc_list, train_ll, val_ll = item_response.irt(boot_datas[1], val_data, lr, irt_iters)
    # print some IRT logs (val_acc_list printed inside irt as well)
    val_acc_irt = item_response.evaluate(val_data, theta, beta)
    test_acc_irt = item_response.evaluate(test_data, theta, beta)
    print(f"IRT - Val acc: {val_acc_irt:.4f} | Test acc: {test_acc_irt:.4f}")

    preds_val_irt = preds_from_irt(theta, beta, val_data)
    preds_test_irt = preds_from_irt(theta, beta, test_data)

    # -------------------------
    # Base model 3: Neural Network (AutoEncoder)
    # -------------------------
    print("\n[BASE 3] Neural Network (train on bootstrap sample #3)")
    # Build bootstrap matrix and zero-imputed version
    boot3_mat = build_matrix_from_interactions(boot_datas[2], num_users, num_questions)
    zero_boot3 = np.nan_to_num(boot3_mat, nan=0.0)  # zeros for missing
    # Convert to torch tensors expected by neural_network.train
    zero_boot3_t = torch.FloatTensor(zero_boot3)
    boot3_t = torch.FloatTensor(boot3_mat)  # contains NaNs, neural_network.train uses mask via isnan

    # Build model
    num_q = num_questions
    ae_k = 100
    model = neural_network.AutoEncoder(num_q, k=ae_k)

    if DEMO:
        nn_epochs = NN_EPOCHS_DEMO
    else:
        nn_epochs = 50

    train_losses, val_accs, best_state = neural_network.train(
        model,
        lr=0.005,
        lamb=0.001,
        train_data=boot3_t,
        zero_train_data=zero_boot3_t,
        valid_data=val_data,
        num_epoch=nn_epochs,
        student_id="ensemble_demo"
    )

    # Prepare best model for prediction
    best_model = neural_network.AutoEncoder(num_q, k=ae_k)
    best_model.load_state_dict(best_state)
    # Evaluate & predict on validation & test
    val_acc_nn = neural_network.evaluate(best_model, zero_boot3_t, val_data)
    test_acc_nn = neural_network.evaluate(best_model, zero_boot3_t, test_data)  # note: evaluate uses train_data param for inputs
    print(f"NN AutoEncoder - Val acc: {val_acc_nn:.4f} | Test acc: {test_acc_nn:.4f}")

    preds_val_nn = preds_from_nn(best_model, zero_boot3_t, val_data)
    preds_test_nn = preds_from_nn(best_model, zero_boot3_t, test_data)

    # -------------------------
    # Ensemble: average three probability vectors
    # -------------------------
    print("\n[ENSEMBLE] Averaging probabilities from KNN, IRT, and NN ...")
    preds_val_stack = np.vstack([preds_val_knn, preds_val_irt, preds_val_nn])  # shape (3, N_val)
    preds_test_stack = np.vstack([preds_test_knn, preds_test_irt, preds_test_nn])

    ensemble_val_probs = preds_val_stack.mean(axis=0)
    ensemble_test_probs = preds_test_stack.mean(axis=0)

    val_acc_ensemble = evaluate_acc_from_probs(ensemble_val_probs, val_data['is_correct'])
    test_acc_ensemble = evaluate_acc_from_probs(ensemble_test_probs, test_data['is_correct'])

    # -------------------------
    # Print final results
    # -------------------------
    print("\n=== FINAL RESULTS ===")
    print(f"KNN   Val/Test: {val_acc_knn:.4f} / {test_acc_knn:.4f}")
    print(f"IRT   Val/Test: {val_acc_irt:.4f} / {test_acc_irt:.4f}")
    print(f"NN    Val/Test: {val_acc_nn:.4f} / {test_acc_nn:.4f}")
    print(f"ENSEMBLE Val/Test: {val_acc_ensemble:.4f} / {test_acc_ensemble:.4f}")
    print("=====================\n")

if __name__ == "__main__":
    main()
