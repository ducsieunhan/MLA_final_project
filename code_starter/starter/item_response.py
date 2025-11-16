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
import matplotlib.pyplot as plt
import time


def sigmoid(x):
    """Apply sigmoid function."""
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
    epsilon = 1e-15 
    
    for i in range(len(data["user_id"])):
        user_id = data["user_id"][i]
        q_id = data["question_id"][i]
        c = data["is_correct"][i]
        
        diff = theta[user_id] - beta[q_id]
        pred_prob = sigmoid(diff)
        pred_prob = np.clip(pred_prob, epsilon, 1. - epsilon)
        log_lklihood += (c * np.log(pred_prob) + 
                         (1. - c) * np.log(1. - pred_prob))
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

    grad_theta = np.zeros_like(theta)
    
    for i in range(len(data["user_id"])):
        user_id = data["user_id"][i]
        q_id = data["question_id"][i]
        c = data["is_correct"][i]
        
        diff = theta[user_id] - beta[q_id] 
        pred_prob = sigmoid(diff)
        error = c - pred_prob
        grad_theta[user_id] += error
        
    theta += lr * grad_theta
    
    grad_beta = np.zeros_like(beta)
    
    for i in range(len(data["user_id"])):
        user_id = data["user_id"][i]
        q_id = data["question_id"][i]
        c = data["is_correct"][i]
        
        diff = theta[user_id] - beta[q_id] 
        pred_prob = sigmoid(diff)
        error = c - pred_prob
        grad_beta[q_id] += -error 
        
    beta += lr * grad_beta
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
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
  
    all_user_ids = data["user_id"] + val_data["user_id"]
    all_q_ids = data["question_id"] + val_data["question_id"]
    num_users = max(all_user_ids) + 1
    num_questions = max(all_q_ids) + 1
    
    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)

    val_acc_lst = []
    train_ll = []
    val_ll = []  
    
    start_time = time.time()

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        
        val_acc_lst.append(score)
        train_ll.append(-neg_lld) 
        val_ll.append(-val_neg_lld)
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Iter {i+1:3d}/{iterations} | Train Loss: {neg_lld:11.4f} | "
                  f"Val Loss: {val_neg_lld:10.4f} | Val Acc: {score:.4f} | "
                  f"Time: {elapsed:.2f}s")
        
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_ll, val_ll


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
        x = (theta[u] - beta[q]).sum()
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
    LEARNING_RATE = 0.01
    ITERATIONS = 300
    STUDENT_ID = "2201040116"  
    
    print(f"Training IRT model with lr={LEARNING_RATE}, iterations={ITERATIONS}")
    
    theta, beta, val_acc_lst, train_ll, val_ll = irt(
        train_data, val_data, 
        LEARNING_RATE, ITERATIONS
    )
    
    print("\n--- Final Evaluation ---")
    
    final_val_acc = evaluate(val_data, theta, beta)
    final_test_acc = evaluate(test_data, theta, beta)
    
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Test Accuracy: {final_test_acc:.4f}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    print("\n--- Generating Plots for Part (d) ---")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    iter_list = list(range(1, len(train_ll) + 1))
    ax1.plot(iter_list, train_ll, label="Training Log-Likelihood", linewidth=2)
    ax1.plot(iter_list, val_ll, label="Validation Log-Likelihood", linewidth=2)
    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_ylabel("Log-Likelihood", fontsize=11)
    ax1.set_title("IRT: Log-Likelihood vs. Iteration", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
  
    sorted_indices = np.argsort(beta)
    q_easy = sorted_indices[len(beta)//4]     
    q_medium = sorted_indices[len(beta)//2]    
    q_hard = sorted_indices[3*len(beta)//4]   
    
    selected_questions = [q_easy, q_medium, q_hard]
    
    theta_range = np.linspace(np.min(theta) - 1, np.max(theta) + 1, 200)
    
   
    subscripts = ['₁', '₂', '₃']
    for idx, q_id in enumerate(selected_questions):
        beta_j = beta[q_id]
        prob_correct = sigmoid(theta_range - beta_j)
        ax2.plot(theta_range, prob_correct, linewidth=2,
                label=f"j{subscripts[idx]} (ID={q_id}, β = {beta_j:.2f})")
    
    ax2.set_xlabel("Student Ability (θ)", fontsize=11)
    ax2.set_ylabel("Probability of Correct Answer", fontsize=11)
    ax2.set_title("IRT: Probability Curves", fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    filename = f"irt_curves_{STUDENT_ID}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {filename}")
    plt.close()
    
  
    print("\n--- Interpretation of Probability Curves ---")
    print(f"Question {q_easy} (Easy, β={beta[q_easy]:.2f}): Low difficulty - students have high")
    print(f"  probability of answering correctly even with lower ability.")
    print(f"Question {q_medium} (Medium, β={beta[q_medium]:.2f}): Medium difficulty - probability")
    print(f"  increases gradually around average student ability.")
    print(f"Question {q_hard} (Hard, β={beta[q_hard]:.2f}): High difficulty - requires higher")
    print(f"  ability to have good chance of answering correctly.")
    print("\nAll curves follow the sigmoid shape, showing smooth transition from low")
    print("to high probability as student ability increases. The horizontal shift")
    print("represents question difficulty (β).")
    
    print("\nCompleted Part A, Question 2.")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()