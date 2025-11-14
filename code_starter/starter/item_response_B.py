# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, [Student Name] ([Student ID]), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.

# This file implements the IRT 2PL model using PyTorch for Part B
# It addresses limitations of the 1PL model (item_response.py) by:
# 1. Upgrading to 2PL (adding alpha/discrimination parameter).
# 2. Using PyTorch for superior optimization (Adam) and autograd.
# 3. Implementing L2 Regularization (weight_decay) to combat overfitting.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class IRT_2PL_Model(nn.Module):
    """
    Implements the 2-Parameter Logistic (2PL) IRT model.
    """
    def __init__(self, num_users, num_questions, l2_lambda=0.0):
        super(IRT_2PL_Model, self).__init__()
        
       
        self.theta = nn.Embedding(num_users, 1)        
        self.beta = nn.Embedding(num_questions, 1)     
        self.alpha = nn.Embedding(num_questions, 1)    

      
        nn.init.normal_(self.theta.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.beta.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.alpha.weight, mean=1.0, std=0.1)

    def forward(self, user_id, question_id):
        """
        Calculates the probability of a correct response.
        P(c_ij=1) = sigmoid(α_j * (θ_i - β_j))
        """
      
        theta_i = self.theta(user_id)         
        beta_j = self.beta(question_id)      
        alpha_j = self.alpha(question_id)    
        
      
        alpha_j_positive = torch.nn.functional.softplus(alpha_j)
        
        diff = alpha_j_positive * (theta_i - beta_j)   
        
        return torch.sigmoid(diff)

    def get_params(self):
        """Helper to get numpy arrays of the parameters for plotting/evaluation."""
        return (
            self.theta.weight.data.cpu().numpy(),
            self.beta.weight.data.cpu().numpy(),
            torch.nn.functional.softplus(self.alpha.weight).data.cpu().numpy()
        )


def load_data_to_tensors(data_dict):
    """Converts data dictionary from utils.py to PyTorch tensors."""
    user_id = torch.tensor(data_dict["user_id"], dtype=torch.long).to(device)
    question_id = torch.tensor(data_dict["question_id"], dtype=torch.long).to(device)
    is_correct = torch.tensor(data_dict["is_correct"], dtype=torch.float).to(device)
    
   
    is_correct = is_correct.view(-1, 1)
    
    return user_id, question_id, is_correct


def train_model(model, train_data, val_data, lr, iterations, l2_lambda, patience):
    """
    Trains the 2PL IRT model using PyTorch.
    """
  
    train_user, train_q, train_correct = load_data_to_tensors(train_data)
    val_user, val_q, val_correct = load_data_to_tensors(val_data)
    
 
    criterion = nn.BCELoss()
    
   
    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=l2_lambda 
    )
    
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    start_time = time.time()
    
    for i in range(iterations):
        model.train() 
        
        optimizer.zero_grad()  
        
        pred_probs = model(train_user, train_q)
        
        loss = criterion(pred_probs, train_correct)
        
       
        loss.backward()
        
  
        optimizer.step()
        
    
        model.eval()  
        with torch.no_grad(): 
            val_probs = model(val_user, val_q)
            val_loss = criterion(val_probs, val_correct)
    
            val_preds = (val_probs >= 0.5).float()
            accuracy = (val_preds == val_correct).float().mean()
            
            val_accuracies.append(accuracy.item())
            val_losses.append(val_loss.item())
            train_losses.append(loss.item())

        elapsed = time.time() - start_time
        

        print(f"Iter {i+1:3d}/{iterations} | "
              f"Train Loss: {loss.item():.4f} | "
              f"Val Loss: {val_loss.item():.4f} | "
              f"Val Acc: {accuracy.item():.4f} | "
              f"Time: {elapsed:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
   
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
    
            print(f"\n--- Early Stopping triggered at iter {i+1} ---")
            print(f"Best Validation Loss: {best_val_loss:.4f}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return train_losses, val_losses, val_accuracies


def evaluate_model(model, data_dict):
    """Evaluates the model and returns accuracy."""
    model.eval() 
    user, q, correct = load_data_to_tensors(data_dict)
    
    with torch.no_grad():
        probs = model(user, q)
        preds = (probs >= 0.5).float()
        accuracy = (preds == correct).float().mean()
    
    return accuracy.item()


def plot_curves(model, student_id):
    """
    Plots the Log-Likelihood and Item Characteristic Curves (ICCs)
    for the improved 2PL model.
    """
    theta, beta, alpha = model.get_params()
    beta = beta.flatten()
    alpha = alpha.flatten()
    
    print("\n--- Generating Plots for Part B (2PL Model) ---")
    
    fig, ax = plt.subplots(figsize=(7, 5))

    sorted_alpha_indices = np.argsort(alpha)
    q_low_alpha = sorted_alpha_indices[len(alpha)//4]
    q_med_alpha = sorted_alpha_indices[len(alpha)//2]
    q_high_alpha = sorted_alpha_indices[3*len(alpha)//4]
    
    selected_questions = [q_low_alpha, q_med_alpha, q_high_alpha]
    
    theta_range = np.linspace(np.min(theta) - 1, np.max(theta) + 1, 200)
    
    subscripts = ['₁', '₂', '₃']
    for idx, q_id in enumerate(selected_questions):
        beta_j = beta[q_id]
        alpha_j = alpha[q_id]
   
        prob_correct = 1 / (1 + np.exp(-alpha_j * (theta_range - beta_j)))
        
        label = (f"j{subscripts[idx]} (α={alpha_j:.2f}, β={beta_j:.2f})")
        ax.plot(theta_range, prob_correct, linewidth=2, label=label)
    
    ax.set_xlabel("Student Ability (θ)", fontsize=11)
    ax.set_ylabel("Probability of Correct Answer", fontsize=11)
    ax.set_title("Part B: 2PL Probability Curves (ICCs)", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    filename = f"part_b_improved_model_{student_id}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {filename}")
    plt.close()
    
    print("\n--- Interpretation of 2PL Curves ---")
    print("Shape: Curves now have DIFFERENT slopes (steepness).")
    print("This is controlled by the alpha (α) parameter:")
    print(f" - High α (e.g., {alpha[q_high_alpha]:.2f}): Steep curve, good at discriminating ability.")
    print(f" - Low α (e.g., {alpha[q_low_alpha]:.2f}): Flat curve, poor at discriminating ability.")
    print(f"The beta (β) parameter still controls the difficulty (horizontal shift).")


def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune these NEW hyperparameters for Part B.                        #
    #####################################################################
    

    num_users = max(
        max(train_data["user_id"]),
        max(val_data["user_id"]),
        max(test_data["user_id"])
    ) + 1
    num_questions = max(
        max(train_data["question_id"]),
        max(val_data["question_id"]),
        max(test_data["question_id"])
    ) + 1
    

    LR = 0.01         
    ITERATIONS = 150  
    L2_LAMBDA = 0.0    
    PATIENCE = 10      
    STUDENT_ID = "2201040116" 
   
    print(f"Training IRT 2PL (PyTorch) model with lr={LR}, "
          f"iterations={ITERATIONS}, l2_lambda={L2_LAMBDA}, patience={PATIENCE}")

    model = IRT_2PL_Model(num_users, num_questions).to(device)


    train_losses, val_losses, val_accuracies = train_model(
        model, train_data, val_data, 
        LR, ITERATIONS, L2_LAMBDA, PATIENCE
    )
    
    print("\n--- Final Evaluation (Using BEST 2PL model) ---")

    final_val_acc = evaluate_model(model, val_data)
    final_test_acc = evaluate_model(model, test_data)
    
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Test Accuracy: {final_test_acc:.4f}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    plot_curves(model, STUDENT_ID)


if __name__ == "__main__":
    main()