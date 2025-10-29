# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, [Student Name] ([Student ID]), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt # Thêm thư viện để vẽ đồ thị


def sigmoid(x):
    """Apply sigmoid function."""
    # Thêm xử lý để tránh tràn số (overflow)
    x = np.clip(x, -500, 500)
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0
    
    # Duyệt qua từng tương tác (i, j, c_ij)
    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        c = data["is_correct"][i]
        
        # Lấy theta_i và beta_j tương ứng
        # Đảm bảo index không vượt quá giới hạn
        if u >= len(theta) or q >= len(beta):
            continue
            
        diff = theta[u] - beta[q]
        
        # Tính theo công thức log-likelihood rút gọn:
        # log_L = c * (theta - beta) - log(1 + exp(theta - beta))
        log_lklihood += c * diff - np.logaddexp(0, diff) # np.logaddexp(0, x) = log(1 + exp(x))
        
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Khởi tạo vector đạo hàm
    theta_grad = np.zeros_like(theta)
    beta_grad = np.zeros_like(beta)
    
    # 1. Tính tổng đạo hàm (Batch Gradient)
    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        c = data["is_correct"][i]

        if u >= len(theta) or q >= len(beta):
            continue

        diff = theta[u] - beta[q]
        p_ij = sigmoid(diff) # Xác suất dự đoán
        
        # Đạo hàm của NLL theo theta[u]: (p_ij - c)
        theta_grad[u] += (p_ij - c)
        
        # Đạo hàm của NLL theo beta[q]: -(p_ij - c)
        beta_grad[q] += -(p_ij - c)
        
    # 2. Cập nhật tham số (Batch Gradient Descent)
    # Lưu ý: Code starter đề cập "alternating gradient descent",
    # nhưng chúng ta có thể cập nhật cả hai cùng lúc (batch GD).
    # Để tuân thủ "alternating", chúng ta sẽ cập nhật theta trước,
    # sau đó dùng theta MỚI để cập nhật beta (hoặc ngược lại).
    # Tuy nhiên, cập nhật đồng thời đơn giản hơn và thường được chấp nhận.
    # Ở đây ta cập nhật đồng thời.
    
    theta = theta - lr * theta_grad
    beta = beta - lr * beta_grad
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst, train_lld_lst)
    """
    # TODO: Initialize theta and beta.
    
    # Tìm số lượng học sinh và câu hỏi tối đa
    # Cần gộp ID từ cả train, valid, (và test) để đảm bảo kích thước
    all_user_ids = np.concatenate((data["user_id"], val_data["user_id"]))
    all_q_ids = np.concatenate((data["question_id"], val_data["question_id"]))
    
    num_students = np.max(all_user_ids) + 1
    num_questions = np.max(all_q_ids) + 1
    
    # Khởi tạo theta và beta bằng 0 (hoặc giá trị ngẫu nhiên nhỏ)
    theta = np.zeros(num_students)
    beta = np.zeros(num_questions)

    val_acc_lst = []
    train_lld_lst = [] # Theo dõi training cost

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        
        val_acc_lst.append(score)
        train_lld_lst.append(neg_lld)
        
        if (i % 10 == 0) or (i == iterations - 1):
            print(f"Epoch {i+1}/{iterations} \t NLLK: {neg_lld:.4f} \t Val Score: {score:.4f}")
            
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # Trả về các giá trị để báo cáo và vẽ đồ thị
    return theta, beta, val_acc_lst, train_lld_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        
        # Xử lý trường hợp ID không có trong tập train
        if u >= len(theta) or q >= len(beta):
            # Nếu không có thông tin, dự đoán 0.5 (False)
            p_a = 0.5
        else:
            x = (theta[u] - beta[q]) # .sum() không cần thiết vì là 1D
            p_a = sigmoid(x)
            
        pred.append(p_a >= 0.5)
        
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    
    # Như đã giải thích ở đầu báo cáo, chúng ta KHÔNG tinh chỉnh 'k'
    # mà sẽ tinh chỉnh 'learning_rate' và 'iterations'.
    
    print("--- Bắt đầu tinh chỉnh siêu tham số cho IRT (lr, iterations) ---")
    
    # 1. Tinh chỉnh siêu tham số
    # Các giá trị để thử nghiệm
    learning_rates = [0.01, 0.005, 0.001]
    num_iterations = [100, 150, 200]

    best_val_acc = -1
    best_lr = 0
    best_iter = 0
    best_model_params = (None, None)
    best_history = (None, None) # (val_accs, train_llds)

    for lr in learning_rates:
        for iters in num_iterations:
            print(f"\n--- Đang huấn luyện với: lr={lr}, iterations={iters} ---")
            theta, beta, val_accs, train_llds = irt(train_data, val_data, lr, iters)
            
            final_val_acc = val_accs[-1]
            print(f"-> Kết quả: Val Acc = {final_val_acc:.4f}")

            if final_val_acc > best_val_acc:
                best_val_acc = final_val_acc
                best_lr = lr
                best_iter = iters
                best_model_params = (theta, beta)
                best_history = (val_accs, train_llds)

    print("\n--- Tinh chỉnh hoàn tất ---")
    print(f"Cấu hình tốt nhất tìm được:")
    print(f"  Learning Rate (lr*): {best_lr}")
    print(f"  Iterations (epoch*): {best_iter}")
    print(f"  Validation Accuracy cao nhất: {best_val_acc:.4f}")

    # 2. Đánh giá trên tập Test với mô hình tốt nhất
    best_theta, best_beta = best_model_params
    
    if best_theta is not None:
        test_acc = evaluate(test_data, best_theta, best_beta)
        print(f"\n=> Độ chính xác trên tập Test (với mô hình tốt nhất): {test_acc:.4f}")

        # 3. Vẽ biểu đồ (cho báo cáo)
        best_val_accs, best_train_llds = best_history
        epochs = range(1, best_iter + 1)
        
        print("\n--- Đang tạo và lưu biểu đồ huấn luyện ---")
        
        # Biểu đồ 1: Training Cost (NLL)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, best_train_llds, label='Training NLL', marker='.')
        plt.xlabel("Epochs")
        plt.ylabel("Negative Log-Likelihood")
        plt.title(f"Training Cost (NLL) vs. Epochs (lr={best_lr})")
        plt.legend()
        plt.grid(True)
        plt.savefig("irt_training_cost.png")
        print("Đã lưu 'irt_training_cost.png'")

        # Biểu đồ 2: Validation Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, best_val_accs, label='Validation Accuracy', marker='.', color='orange')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(f"Validation Accuracy vs. Epochs (lr={best_lr})")
        plt.legend()
        plt.grid(True)
        plt.savefig("irt_validation_accuracy.png")
        print("Đã lưu 'irt_validation_accuracy.png'")
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    # Phần (d) thường yêu cầu vẽ đường cong đặc tính câu hỏi (ICC)
    # để thể hiện mối quan hệ giữa năng lực (theta) và xác suất trả lời đúng.
    
    print("\n--- Thực hiện Phần (d): Vẽ đường cong đặc tính câu hỏi (ICC) ---")
    
    if best_beta is not None:
        # Sắp xếp các câu hỏi dựa trên độ khó (beta)
        sorted_beta_indices = np.argsort(best_beta)
        
        # Lấy các câu hỏi có nhiều tương tác (để beta đáng tin cậy)
        # (Bỏ qua bước này để đơn giản hóa, chỉ lấy dễ/TB/khó)
        
        # Chọn 3 câu hỏi: dễ nhất, khó nhất, và ở giữa
        q_easy_id = sorted_beta_indices[0]
        q_medium_id = sorted_beta_indices[len(sorted_beta_indices) // 2]
        q_hard_id = sorted_beta_indices[-1]
        
        beta_easy = best_beta[q_easy_id]
        beta_medium = best_beta[q_medium_id]
        beta_hard = best_beta[q_hard_id]

        print(f"Câu dễ (q_id={q_easy_id}): beta = {beta_easy:.4f}")
        print(f"Câu TB (q_id={q_medium_id}): beta = {beta_medium:.4f}")
        print(f"Câu khó (q_id={q_hard_id}): beta = {beta_hard:.4f}")

        # Tạo một dải năng lực học sinh (theta) để vẽ
        theta_range = np.linspace(-5, 5, 200)
        
        # Tính xác suất trả lời đúng cho từng câu hỏi
        prob_easy = sigmoid(theta_range - beta_easy)
        prob_medium = sigmoid(theta_range - beta_medium)
        prob_hard = sigmoid(theta_range - beta_hard)
        
        # Vẽ đồ thị
        plt.figure(figsize=(10, 6))
        plt.plot(theta_range, prob_easy, label=f"Câu dễ (q={q_easy_id}, $\\beta$={beta_easy:.2f})")
        plt.plot(theta_range, prob_medium, label=f"Câu TB (q={q_medium_id}, $\\beta$={beta_medium:.2f})")
        plt.plot(theta_range, prob_hard, label=f"Câu khó (q={q_hard_id}, $\\beta$={beta_hard:.2f})")
        
        plt.title("Đường cong đặc tính câu hỏi (Item Characteristic Curves - ICC)")
        plt.xlabel("Năng lực học sinh ($\\theta$)")
        plt.ylabel("Xác suất trả lời đúng P(c=1)")
        plt.axvline(x=0, color='gray', linestyle='--', label='Năng lực TB ($\\theta=0$)')
        plt.legend()
        plt.grid(True)
        plt.savefig("irt_icc_plot.png")
        print("Đã lưu 'irt_icc_plot.png'")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()