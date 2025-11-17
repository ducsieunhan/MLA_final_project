# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, Pham Thi Khanh Ngoc (2201040133), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import load_valid_csv, load_public_test_csv, load_train_sparse

def load_data(base_path="./data"):
    """
    Load the data and return: zero_train_matrix, train_matrix, valid_data, test_data.
    zero_train_matrix: missing entries filled with 0 (for input).
    train_matrix: preserves NaNs (for masking).
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)
    return zero_train_matrix, train_matrix, valid_data, test_data

class AutoEncoder(nn.Module):
    """
    Simple autoencoder for student response prediction.
    - Input: response vector (missing as 0)
    - Encoder: Linear + sigmoid
    - Decoder: Linear + sigmoid
    """
    def __init__(self, num_question, k=100):
        super(AutoEncoder, self).__init__()
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)
        self.sigmoid = nn.Sigmoid()

        # Xavier init for stability
        nn.init.xavier_uniform_(self.g.weight)
        nn.init.zeros_(self.g.bias)
        nn.init.xavier_uniform_(self.h.weight)
        nn.init.zeros_(self.h.bias)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2 for regularization."""
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """
        TODO:
        - Implement the forward pass:
          out = sigmoid(h(sigmoid(g(inputs))))
        - Return the output vector.
        """
        # === YOUR CODE HERE ===
        z = self.sigmoid(self.g(inputs))
        out = self.sigmoid(self.h(z))
        return out

def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, student_id=""):
    """
    Train the autoencoder with L2 regularization.
    TODO:
    - For each epoch, for each user:
        * Compute forward pass.
        * Compute masked squared error loss for observed entries only.
        * Add L2 regularization (lamb * model.get_weight_norm()).
        * Backprop and optimizer step.
    - Track/plot training loss and validation accuracy.
    - Save plot as autoencoder_results_{student_id}.png.
    - Print summary with best k/lambda/validation accuracy.
    """
    # === YOUR CODE HERE ===
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_users = zero_train_data.shape[0]
    train_losses = []
    valid_accuracies = []
    best_val, best_state = -1.0, None

    for epoch in range(1, num_epoch + 1):
        model.train()
        epoch_loss = 0.0

        for u in range(n_users):
            # inputs = Variable(zero_train_data[u]).unsqueeze(0).to(device)  # zero-filled input
            # target = Variable(train_data[u]).unsqueeze(0).to(device)       # contains NaNs for masking

            inputs = Variable(zero_train_data[u]).unsqueeze(0)  # CPU
            target = Variable(train_data[u]).unsqueeze(0)  # CPU

            output = model(inputs)

            mask = ~torch.isnan(target)
            mse = ((output - target)[mask] ** 2).mean()

            reg = 0.5 * lamb * model.get_weight_norm()
            loss = mse + reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Track metrics
        avg_loss = epoch_loss / n_users
        train_losses.append(avg_loss)

        # Evaluate (CPU)
        val_acc = evaluate(model, zero_train_data, valid_data)
        valid_accuracies.append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:03d}/{num_epoch} | loss={avg_loss:.6f} | val_acc={val_acc:.4f}")


    if student_id:
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(valid_accuracies) + 1), valid_accuracies, label="Valid Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        out_png = f"autoencoder_results_{student_id}.png"
        plt.savefig(out_png, dpi=160)
        plt.close()
        print(f"[train] Saved training curves to {out_png}")

    return train_losses, valid_accuracies, best_state



def evaluate(model, train_data, valid_data):
    """
    Evaluate the model on valid_data. (Already provided.)
    """
    model.eval()
    total = 0
    correct = 0
    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)
        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    num_question = train_matrix.shape[1]

    #####################################################################
    # TODO:
    # 1. Try at least 5 values of k; select best k via validation set.
    # 2. Tune lr, lamb, num_epoch.
    # 3. Train AutoEncoder, plot/save learning curves, validation accuracy.
    # 4. Report best k and corresponding metrics.
    # 5. Save plot as autoencoder_results_{student_id}.png.
    # 6. Write a reflection on regularization and k in your report.
    #####################################################################

    # Example (students must replace with their chosen values)
    student_id = "2201040133"  # TODO: fill with your real student ID
    k_list = [10, 50, 100, 200, 500]
    lr_list = [5e-3, 1e-3]
    lamb_list = [1e-3, 3e-4]
    epoch_list  = [30, 50]

    best = {"val": -1.0}
    best_hist = None  # (train_losses, val_accs)

    per_k_best = {k: -1.0 for k in k_list}

    for k in k_list:
        for lr in lr_list:
            for lamb in lamb_list:
                for num_epoch in epoch_list:
                    print("=" * 80)
                    print(f"Training AutoEncoder(k={k}, lr={lr}, lamb={lamb}, epochs={num_epoch})")
                    model = AutoEncoder(num_question, k)

                    init_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}

                    train_losses, val_accs, best_state = train(
                        model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, student_id=""
                    )

                    # Validation accuracy tốt nhất trong các epoch của cấu hình hiện tại
                    best_val_cfg = float(max(val_accs))

                    # Cập nhật "best theo k"
                    if best_val_cfg > per_k_best[k]:
                        per_k_best[k] = best_val_cfg

                    # Cập nhật best toàn cục (k*, lr*, lamb*, epochs*)
                    if best_val_cfg > best["val"]:
                        best = {
                            "val": best_val_cfg,
                            "k": k, "lr": lr, "lamb": lamb, "epochs": num_epoch,
                            "state": best_state,
                            "init_state": init_state
                        }
                        best_hist = (train_losses, val_accs)

    # (c):
    # Validation Accuracy cho k và k*
    print("#" * 80)
    print("Validation accuracy for each k (best over lr, lambda, epochs):")
    for k in k_list:
        print(f"k={k:>3} -> val_acc={per_k_best[k]:.4f}")

    # k*
    k_star = max(per_k_best, key=lambda kk: per_k_best[kk])
    print("#" * 80)
    print(f"Selected k* = {k_star} with highest validation accuracy = {per_k_best[k_star]:.4f}")
    print(f"BEST CONFIG (by validation accuracy): k*={best['k']}, lr*={best['lr']}, lamb*={best['lamb']}, epochs*={best['epochs']}")


    #(d)
    print(f"[Retrain-best] k={best['k']}, lr={best['lr']}, lamb={best['lamb']}, epochs={best['epochs']}")
    torch.manual_seed(0)
    np.random.seed(0)
    best_model_for_plot = AutoEncoder(num_question, best["k"])
    best_model_for_plot.load_state_dict(best["init_state"])
    _ = train(
        best_model_for_plot,
        best["lr"], best["lamb"],
        train_matrix, zero_train_matrix, valid_data,
        best["epochs"],
        student_id=student_id
    )

    # After training, evaluate on validation and/or test data as required.

    #(e)
    best_model = AutoEncoder(num_question, best["k"])  # CPU
    best_model.load_state_dict(best["state"])
    final_test_acc = evaluate(best_model, zero_train_matrix, test_data)
    print(f"Final TEST accuracy (with best config): {final_test_acc:.4f}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

if __name__ == "__main__":
    main()
