# item_response.py
# ĐÃ CẬP NHẬT: Thêm kỹ thuật Early Stopping để chống overfitting.

import numpy as np
import matplotlib.pyplot as plt
import utils  # Import file utils.py do giảng viên cung cấp
from typing import List, Dict, Tuple
import time # Thêm thư viện time để theo dõi

# --- Helper Functions (Không thay đổi) ---

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Tính hàm sigmoid."""
    # Thêm xử lý chống tràn (overflow) cho các giá trị quá lớn/nhỏ
    x = np.clip(x, -500, 500)
    return 1. / (1. + np.exp(-x))

def convert_data_format(data_dict: Dict[str, List]) -> List[Dict]:
    """Chuyển đổi định dạng dữ liệu (Giữ nguyên)."""
    list_of_dicts = []
    if not data_dict["user_id"]:
         return []
    num_records = len(data_dict["user_id"])
    has_correct = "is_correct" in data_dict and len(data_dict["is_correct"]) == num_records
    for i in range(num_records):
        record = {
            "user_id": data_dict["user_id"][i],
            "question_id": data_dict["question_id"][i]
        }
        if has_correct:
            record["is_correct"] = data_dict["is_correct"][i]
        list_of_dicts.append(record)
    return list_of_dicts

def neg_log_likelihood(data_list: List[Dict], 
                       theta: np.ndarray, 
                       beta: np.ndarray) -> float:
    """Tính negative log-likelihood (NLL) (Giữ nguyên)."""
    nll = 0.
    epsilon = 1e-15
    for record in data_list:
        if "is_correct" not in record:
            continue
        user_id = record["user_id"]
        q_id = record["question_id"]
        c = record["is_correct"]
        
        diff = theta[user_id] - beta[q_id]
        pred_prob = sigmoid(diff)
        pred_prob = np.clip(pred_prob, epsilon, 1. - epsilon)
        nll += -(c * np.log(pred_prob) + (1. - c) * np.log(1. - pred_prob))
    return nll

def predict_probabilities(data_list: List[Dict], 
                          theta: np.ndarray, 
                          beta: np.ndarray) -> np.ndarray:
    """Dự đoán xác suất (Giữ nguyên)."""
    preds = []
    for record in data_list:
        user_id = record["user_id"]
        q_id = record["question_id"]
        diff = theta[user_id] - beta[q_id]
        pred_prob = sigmoid(diff)
        preds.append(pred_prob)
    return np.array(preds)

# --- Core Training Function (ĐÃ NÂNG CẤP) ---

def irt_train(train_list: List[Dict], 
              val_list: List[Dict],
              val_dict: Dict[str, List],
              num_users: int, 
              num_questions: int, 
              lr: float, 
              iterations: int,
              patience: int = 5) -> Tuple[np.ndarray, np.ndarray, List, List, int]:
    """
    Train mô hình IRT với EARLY STOPPING.
    
    Args:
        (Các tham số cũ)
        iterations (int): Số vòng lặp TỐI ĐA.
        patience (int): Số vòng lặp kiên nhẫn chờ cải thiện trước khi dừng.
        
    Returns:
        Tuple chứa (theta_tốt_nhất, beta_tốt_nhất, train_ll, val_ll, iter_tốt_nhất)
    """
    # Khởi tạo tham số
    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)
    
    # Biến cho Early Stopping
    best_val_ll = -np.inf  # Ta muốn tối đa hóa LL
    best_theta = np.copy(theta)
    best_beta = np.copy(beta)
    best_iter = 0
    patience_counter = 0
    
    train_ll_history = []
    val_ll_history = []
    
    start_time = time.time()
    
    print(f"Bắt đầu training (tối đa {iterations} vòng, patience={patience}):")
    
    for i in range(iterations):
        # Tính gradient (Giữ nguyên)
        grad_theta = np.zeros_like(theta)
        grad_beta = np.zeros_like(beta)
        
        for record in train_list:
            user_id = record["user_id"]
            q_id = record["question_id"]
            c = record["is_correct"]
            diff = theta[user_id] - beta[q_id]
            pred_prob = sigmoid(diff)
            error = c - pred_prob
            grad_theta[user_id] += error
            grad_beta[q_id] += -error
            
        # Cập nhật tham số (Gradient Ascent)
        theta += lr * grad_theta
        beta += lr * grad_beta
        
        # Ghi lại log-likelihood
        train_nll = neg_log_likelihood(train_list, theta, beta)
        val_nll = neg_log_likelihood(val_list, theta, beta)
        train_ll = -train_nll
        val_ll = -val_nll
        train_ll_history.append(train_ll)
        val_ll_history.append(val_ll)
        
        # --- LOGIC EARLY STOPPING ---
        if val_ll > best_val_ll:
            # Cải thiện: Lưu lại mô hình
            best_val_ll = val_ll
            best_theta = np.copy(theta)
            best_beta = np.copy(beta)
            best_iter = i
            patience_counter = 0
        else:
            # Không cải thiện: Tăng bộ đếm kiên nhẫn
            patience_counter += 1
            
        if (i + 1) % 10 == 0:
            val_preds = predict_probabilities(val_list, theta, beta)
            val_acc = utils.evaluate(val_dict, val_preds)
            elapsed = time.time() - start_time
            print(f"Iter {i+1:3d}/{iterations} | Train LL: {train_ll:11.4f} | "
                  f"Val LL: {val_ll:10.4f} (Best: {best_val_ll:10.4f} at iter {best_iter+1}) | "
                  f"Val Acc: {val_acc:.4f} | Patience: {patience_counter}/{patience} | Time: {elapsed:.2f}s")
        
        # Kiểm tra điều kiện dừng
        if patience_counter >= patience:
            print(f"\n--- Dừng sớm (Early Stopping) tại vòng {i + 1} ---")
            print(f"Mô hình tốt nhất được tìm thấy tại vòng {best_iter + 1} "
                  f"với Val LL: {best_val_ll:.4f}")
            break
            
    if i == iterations - 1:
        print("\n--- Training hoàn tất (đạt max iterations) ---")
        print(f"Mô hình tốt nhất được tìm thấy tại vòng {best_iter + 1} "
              f"với Val LL: {best_val_ll:.4f}")

    return best_theta, best_beta, train_ll_history, val_ll_history, best_iter

# --- Plotting Functions (Giữ nguyên) ---

def plot_log_likelihood(iter_list: List[int], 
                        train_ll: List[float], 
                        val_ll: List[float], 
                        student_id: str,
                        best_iter: int = -1):
    """Vẽ biểu đồ Log-Likelihood."""
    filename = f"irt_ll_curves_{student_id}.png"
    plt.figure(figsize=(10, 6))
    plt.plot(iter_list, train_ll, label="Training Log-Likelihood")
    plt.plot(iter_list, val_ll, label="Validation Log-Likelihood")
    
    # Thêm một đường dọc đánh dấu điểm dừng sớm
    if best_iter >= 0:
        plt.axvline(x=best_iter + 1, color='r', linestyle='--', 
                    label=f'Best Model (Iter {best_iter + 1})')
        
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.title("IRT: Log-Likelihood vs. Iteration (with Early Stopping)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"Đã lưu biểu đồ Log-Likelihood vào: {filename}")
    plt.close()

def plot_probability_curves(theta: np.ndarray, 
                            beta: np.ndarray, 
                            question_ids: List[int], 
                            student_id: str):
    """Vẽ biểu đồ xác suất."""
    filename = f"irt_prob_curves_{student_id}.png"
    plt.figure(figsize=(10, 6))
    theta_range = np.linspace(np.min(theta) - 1, np.max(theta) + 1, 200)
    for q_id in question_ids:
        beta_j = beta[q_id]
        prob_correct = sigmoid(theta_range - beta_j)
        plt.plot(theta_range, prob_correct, 
                 label=f"Question {q_id} (β = {beta_j:.2f})")
    plt.xlabel("Student Ability (θ)")
    plt.ylabel("Probability of Correct Answer P(c=1)")
    plt.title("IRT: Probability Curves (from Best Model)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"Đã lưu biểu đồ Probability Curves vào: {filename}")
    plt.close()

# --- Main execution (ĐÃ CẬP NHẬT) ---

if __name__ == "__main__":
    
    # --- 1. Thiết lập Hyperparameters ---
    LEARNING_RATE = 0.01  # Giữ nguyên lr
    MAX_ITERATIONS = 300  # Tăng số vòng lặp tối đa
    EARLY_STOP_PATIENCE = 5 # Dừng nếu Val LL không cải thiện sau 5 vòng
    STUDENT_ID = "2201040116"  # <<< THAY THẾ BẰNG ID CỦA BẠN
    DATA_DIR = "./data" 
    
    print("Đang tải dữ liệu...")
    train_data_dict = utils.load_train_csv(root_dir=DATA_DIR)
    val_data_dict = utils.load_valid_csv(root_dir=DATA_DIR)
    test_data_dict = utils.load_public_test_csv(root_dir=DATA_DIR)
    
    if not train_data_dict["user_id"]:
        print(f"Không tìm thấy dữ liệu training tại {DATA_DIR}. Thoát.")
        exit()

    train_data_list = convert_data_format(train_data_dict)
    val_data_list = convert_data_format(val_data_dict)
    test_data_list = convert_data_format(test_data_dict)
    
    all_user_ids = train_data_dict["user_id"] + val_data_dict["user_id"] + test_data_dict["user_id"]
    all_q_ids = train_data_dict["question_id"] + val_data_dict["question_id"] + test_data_dict["question_id"]
    NUM_USERS = max(all_user_ids) + 1
    NUM_QUESTIONS = max(all_q_ids) + 1
    
    print(f"Tổng số User: {NUM_USERS}, Tổng số Question: {NUM_QUESTIONS}")
    
    # --- 2. Training (với Early Stopping) ---
    # Hàm train giờ trả về tham số TỐT NHẤT
    theta, beta, train_ll, val_ll, best_iter = irt_train(
        train_data_list, val_data_list, val_data_dict,
        NUM_USERS, NUM_QUESTIONS, 
        LEARNING_RATE, MAX_ITERATIONS, EARLY_STOP_PATIENCE
    )
    
    # --- 3. Đánh giá (Sử dụng tham số TỐT NHẤT) ---
    print("\n--- Kết quả đánh giá (từ mô hình tốt nhất) ---")
    
    val_preds = predict_probabilities(val_data_list, theta, beta)
    final_val_acc = utils.evaluate(val_data_dict, val_preds)
    
    test_preds = predict_probabilities(test_data_list, theta, beta)
    final_test_acc = utils.evaluate(test_data_dict, test_preds)
    
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Test Accuracy: {final_test_acc:.4f}")

    # --- 4. Vẽ biểu đồ (Sử dụng tham số TỐT NHẤT) ---
    print("\nĐang tạo biểu đồ...")
    
    iter_list = list(range(1, len(train_ll) + 1))
    plot_log_likelihood(iter_list, train_ll, val_ll, STUDENT_ID, best_iter)
    
    selected_questions = [0, 1, 2] 
    if NUM_QUESTIONS < 3:
        selected_questions = list(range(NUM_QUESTIONS))
    plot_probability_curves(theta, beta, selected_questions, STUDENT_ID)
    
    print("\nHoàn thành Part A, Bài 2.")