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
    sparse_matrix_evaluate,  # Hàm tiện ích để đánh giá
)


def user_knn_predict_hanu(matrix, data, k, return_confusion=False):
    """
    Dự đoán các giá trị còn thiếu bằng KNN dựa trên người dùng (user-based).
    Args:
        matrix: 2D numpy array (users x questions) với NaNs cho các giá trị còn thiếu.
        data: dict chứa user_id, question_id, is_correct (ví dụ: val_data, test_data).
        k: int, số lượng hàng xóm.
        return_confusion: bool, nếu True, trả về (accuracy, confusion_matrix, predictions).
                          nếu False, chỉ trả về accuracy.
    Returns:
        accuracy: float
        (optional) confusion_matrix: ma trận nhầm lẫn 2x2
        (optional) predictions: mảng numpy chứa các giá trị dự đoán (xác suất)
    """
    imputer = KNNImputer(n_neighbors=k, metric='nan_euclidean')

    imputed_matrix = imputer.fit_transform(matrix)

    acc = sparse_matrix_evaluate(data, imputed_matrix)
    if return_confusion:

        y_true = []
        y_pred_values = []
        y_pred_labels = []
        for i in range(len(data["user_id"])):
            user_id = data["user_id"][i]
            question_id = data["question_id"][i]

            true_label = data["is_correct"][i]
            y_true.append(true_label)

            pred_value = imputed_matrix[user_id, question_id]
            y_pred_values.append(pred_value)

            pred_label = 1 if pred_value >= 0.5 else 0
            y_pred_labels.append(pred_label)

        cm = confusion_matrix(y_true, y_pred_labels)
        return acc, cm, np.array(y_pred_values)
    return acc


def item_knn_predict_hanu(matrix, data, k):
    """
    Dự đoán các giá trị còn thiếu bằng KNN dựa trên item (item-based).
    Args:
        matrix: 2D numpy array (users x questions) với NaNs.
        data: dict chứa user_id, question_id, is_correct (ví dụ: val_data).
        k: int, số lượng hàng xóm.
    Returns:
        accuracy: float
        predictions: mảng numpy chứa các giá trị dự đoán (xác suất)
    """

    matrix_transposed = matrix.T

    imputer = KNNImputer(n_neighbors=k, metric='nan_euclidean')

    imputed_matrix_transposed = imputer.fit_transform(matrix_transposed)

    imputed_matrix = imputed_matrix_transposed.T

    acc = sparse_matrix_evaluate(data, imputed_matrix)

    y_pred_values = []
    for i in range(len(data["user_id"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        pred_value = imputed_matrix[user_id, question_id]
        y_pred_values.append(pred_value)
    return acc, np.array(y_pred_values)


def main():

    sparse_matrix = load_train_sparse("code_starter/starter/data").toarray()
    val_data = load_valid_csv("code_starter/starter/data")
    test_data = load_public_test_csv("code_starter/starter/data")

    val_true_labels = np.array(val_data["is_correct"])

    print(f"Shape của ma trận: {sparse_matrix.shape}") # (542 users, 1774 questions)

    print("\n" + "="*30)
    print("Bắt đầu thử nghiệm User-based KNN...")
    print("="*30)
    k_values = [1, 6, 11, 16, 21, 26]
    user_accuracies = []
    best_user_acc = -1
    best_user_k = -1
    best_user_preds = None
    for k in k_values:
        acc, cm, preds = user_knn_predict_hanu(sparse_matrix, val_data, k, return_confusion=True)
        user_accuracies.append(acc)
        print(f"\n[User-based, k={k}]")
        print(f"Validation Accuracy: {acc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        # lưu
        if acc > best_user_acc:
            best_user_acc = acc
            best_user_k = k
            best_user_preds = preds

    print("\n" + "="*30)
    print("Bắt đầu thử nghiệm Item-based KNN...")
    print("="*30)
    item_accuracies = []
    best_item_acc = -1
    best_item_k = -1
    best_item_preds = None

    STUDENT_ID = "2201040051"
    for k in k_values:

        acc, preds = item_knn_predict_hanu(sparse_matrix, val_data, k)
        item_accuracies.append(acc)
        print(f"\n[Item-based, k={k}]")
        print(f"Validation Accuracy: {acc:.4f}")
        # lưu kết quả tốt nhất
        if acc > best_item_acc:
            best_item_acc = acc
            best_item_k = k
            best_item_preds = preds

    if best_item_preds is not None:
        filename = f"{STUDENT_ID}_item_knn_preds.npy"
        try:
            np.save(filename, best_item_preds)
            print(f"\nĐã lưu dự đoán Item-based (k={best_item_k}) vào file: {filename}")
        except Exception as e:
            print(f"\nLỗi khi lưu file: {e}")

    # ROC-AUC
    print("\n" + "="*30)
    print("Báo cáo ROC-AUC (cho k tốt nhất trên tập Validation)")
    print("="*30)
    if best_user_preds is not None:
        user_roc_auc = roc_auc_score(val_true_labels, best_user_preds)
        print(f"User-based (k={best_user_k}) - ROC-AUC: {user_roc_auc:.4f}")
    if best_item_preds is not None:
        item_roc_auc = roc_auc_score(val_true_labels, best_item_preds)
        print(f"Item-based (k={best_item_k}) - ROC-AUC: {item_roc_auc:.4f}")

    print("\n" + "="*30)
    print("Tổng kết và Báo cáo Test Accuracy")
    print("="*30)

    best_k_final = -1
    best_acc_final = -1
    reflection_statement = ""

    if best_user_acc >= best_item_acc:
        print(f"Mô hình tốt nhất: User-based (k={best_user_k}) với Val Acc = {best_user_acc:.4f}")
        print("Đang chạy trên tập Test...")

        test_acc = user_knn_predict_hanu(sparse_matrix, test_data, best_user_k, return_confusion=False)
        best_k_final = best_user_k
        best_acc_final = test_acc
        reflection_statement = (
            f"User-based KNN cho kết quả tốt nhất khi k={best_user_k}. "
            "Nhìn chung, độ chính xác tăng khi k tăng từ 1, đạt đỉnh, sau đó giảm nhẹ, "
            "cho thấy k quá nhỏ (k=1) bị nhiễu và k quá lớn làm mờ đi các hàng xóm thực sự 'gần'."
        )
    else:
        print(f"Mô hình tốt nhất: Item-based (k={best_item_k}) với Val Acc = {best_item_acc:.4f}")
        print("Đang chạy trên tập Test...")

        test_acc, _ = item_knn_predict_hanu(sparse_matrix, test_data, best_item_k)
        best_k_final = best_item_k
        best_acc_final = test_acc
        reflection_statement = (
            f"Item-based KNN cho kết quả tốt nhất khi k={best_item_k}. "
            "Phương pháp này hoạt động tốt hơn user-based, có thể vì việc 'giống nhau' giữa các câu hỏi "
            "là một chỉ báo mạnh hơn là sự 'giống nhau' giữa các học sinh."
        )

    print("\n--- KẾT QUẢ CUỐI CÙNG (PART A.1) ---")
    print(f"[Summary] For K={best_k_final}, mô hình KNN tốt nhất ({"User" if best_user_acc >= best_item_acc else "Item"}-based) "
          f"đạt được {best_acc_final:.4f} test accuracy.")
    print("\nReflection (Yêu cầu 1d - Hãy chỉnh sửa lại bằng từ ngữ của riêng bạn):")
    print(reflection_statement)


if __name__ == "__main__":
    main()