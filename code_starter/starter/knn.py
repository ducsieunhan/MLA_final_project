# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, [Student Name] ([Student ID]), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.

import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, roc_auc_score
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def user_knn_predict_hanu(matrix, valid_data, k, return_confusion=False):
    """
    Predict missing values using user-based k-nearest neighbors (KNN).
    Args:
        matrix: 2D numpy array (users x questions) with NaNs for missing.
        valid_data: dict with user_id, question_id, is_correct
        k: int, number of neighbors.
        return_confusion: bool, if True also return confusion matrix.
    Returns:
        accuracy: float
        (optional) confusion_matrix: 2x2 confusion matrix
        (optional) predictions: numpy array of predicted values (probabilities)
    """
    imputer = KNNImputer(n_neighbors=k, metric='nan_euclidean')
    imputed_matrix = imputer.fit_transform(matrix)

    acc = sparse_matrix_evaluate(valid_data, imputed_matrix)
    if return_confusion:
        y_true = []
        y_pred_values = []
        y_pred_labels = []
        for i in range(len(valid_data["user_id"])):
            user_id = valid_data["user_id"][i]
            question_id = valid_data["question_id"][i]
            true_label = valid_data["is_correct"][i]
            y_true.append(true_label)

            pred_value = imputed_matrix[user_id, question_id]
            y_pred_values.append(pred_value)

            pred_label = 1 if pred_value >= 0.5 else 0
            y_pred_labels.append(pred_label)

        cm = confusion_matrix(y_true, y_pred_labels)
        return acc, cm, np.array(y_pred_values)
    return acc


def item_knn_predict_hanu(matrix, valid_data, k, student_id=""):
    """
    Predict missing values using item-based k-nearest neighbors (KNN).
    Also saves validation predictions to file named '{student_id}_item_knn_preds.npy'

    Args:
        matrix: 2D numpy array (users x questions) with NaNs for missing.
        valid_data: dict with user_id, question_id, is_correct.
        k: int, number of neighbors.
        student_id: str, student ID (if provided, save predictions).
    Returns:
        accuracy: float
    """
    matrix_transposed = matrix.T
    imputer = KNNImputer(n_neighbors=k, metric='nan_euclidean')
    imputed_matrix_transposed = imputer.fit_transform(matrix_transposed)
    imputed_matrix = imputed_matrix_transposed.T

    acc = sparse_matrix_evaluate(valid_data, imputed_matrix)

    if student_id:
        y_pred_values = []
        for i in range(len(valid_data["user_id"])):
            user_id = valid_data["user_id"][i]
            question_id = valid_data["question_id"][i]
            pred_value = imputed_matrix[user_id, question_id]
            y_pred_values.append(pred_value)
        filename = f"{student_id}_item_knn_preds.npy"
        try:
            np.save(filename, np.array(y_pred_values))
            print(f"  → Saved predictions to: {filename}")
        except Exception as e:
            print(f"  → Error saving file: {e}")

    return acc


def main():
    np.random.seed(42)

    DATA_PATH = "./data"
    try:
        sparse_matrix = load_train_sparse(DATA_PATH).toarray()
        val_data = load_valid_csv(DATA_PATH)
        test_data = load_public_test_csv(DATA_PATH)
    except Exception as e:
        print(f"Error: Could not find data at '{DATA_PATH}'.")
        print("Please ensure you are running this script from the 'starter/' directory.")
        return

    print(f"Matrix shape: {sparse_matrix.shape}")
    print(f"Number of users: {sparse_matrix.shape[0]}")
    print(f"Number of questions: {sparse_matrix.shape[1]}")
    val_true_labels = np.array(val_data["is_correct"])

    print("\n" + "="*60)
    print("Part (a): User-based KNN")
    print("="*60)
    k_values = [1, 6, 11, 16, 21, 26]
    user_results = []
    best_user_acc = -1
    best_user_k = -1
    best_user_preds = None

    for k in k_values:
        acc, cm, preds = user_knn_predict_hanu(sparse_matrix, val_data, k, return_confusion=True)
        user_results.append((k, acc, cm, preds))
        print(f"\n[User-based KNN, k={k}]")
        print(f"Validation Accuracy: {acc:.4f}")
        print(f"Confusion Matrix:")
        print(cm)
        print(f"  TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

        if acc > best_user_acc:
            best_user_acc = acc
            best_user_k = k
            best_user_preds = preds

    print(f"\n{'='*60}")
    print(f"Best User-based KNN: k={best_user_k} with accuracy={best_user_acc:.4f}")
    print(f"{'='*60}")

    print("\n" + "="*60)
    print("Part (b): Item-based KNN")
    print("="*60)

    STUDENT_ID = "2201040051"
    item_results = []
    best_item_acc = -1
    best_item_k = -1
    best_item_preds = None

    for k in k_values:
        acc = item_knn_predict_hanu(sparse_matrix, val_data, k, student_id="")
        item_results.append((k, acc))

        print(f"\n[Item-based KNN, k={k}]")
        print(f"Validation Accuracy: {acc:.4f}")

        if acc > best_item_acc:
            best_item_acc = acc
            best_item_k = k

    print(f"\n{'='*60}")
    print(f"Best Item-based KNN: k={best_item_k} with accuracy={best_item_acc:.4f}")
    print(f"{'='*60}")

    print(f"\nSaving predictions for best Item-based model (k={best_item_k})...")
    _ = item_knn_predict_hanu(sparse_matrix, val_data, best_item_k, STUDENT_ID)

    try:
        best_item_preds = np.load(f"{STUDENT_ID}_item_knn_preds.npy")
    except Exception as e:
        print(f"Warning: Could not load predictions file for ROC-AUC: {e}")

    print("\n" + "="*60)
    print("Part (c): ROC-AUC Evaluation (on Validation Set)")
    print("="*60)

    if best_user_preds is not None:
        user_roc_auc = roc_auc_score(val_true_labels, best_user_preds)
        print(f"User-based (k={best_user_k}) - ROC-AUC: {user_roc_auc:.4f}")

    if best_item_preds is not None:
        item_roc_auc = roc_auc_score(val_true_labels, best_item_preds)
        print(f"Item-based (k={best_item_k}) - ROC-AUC: {item_roc_auc:.4f}")

    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)

    if best_user_acc >= best_item_acc:
        print(f"\nBest Overall Model: User-based (k={best_user_k})")
        print(f"Validation Accuracy: {best_user_acc:.4f}")
        print("Evaluating on test set...")
        test_acc = user_knn_predict_hanu(sparse_matrix, test_data, best_user_k, return_confusion=False)
        best_k = best_user_k
        model_type = "User-based"
    else:
        print(f"\nBest Overall Model: Item-based (k={best_item_k})")
        print(f"Validation Accuracy: {best_item_acc:.4f}")
        print("Evaluating on test set...")
        test_acc = item_knn_predict_hanu(sparse_matrix, test_data, best_item_k, student_id="")
        best_k = best_item_k
        model_type = "Item-based"

    print(f"Test Accuracy: {test_acc:.4f}")

    print("\n" + "="*60)
    print("Part (d): Summary and Reflection")
    print("="*60)

    print(f"\n[Summary] For K={best_k}, the {model_type} KNN achieved {test_acc:.4f} test accuracy.")


if __name__ == "__main__":
    main()