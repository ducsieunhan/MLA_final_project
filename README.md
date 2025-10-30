# Bài tập Cuối kỳ: Machine Learning 

Dự án này triển khai các thuật toán học máy để dự đoán khả năng trả lời đúng của sinh viên, dựa trên dữ liệu từ nền tảng giáo dục Eedi.


## 📂 Cấu trúc thư mục

```
mla_final_project/
|-- data/                 # Chứa file .csv và .npz
|-- src/                  # Chứa toàn bộ code .py
|   |-- utils.py            # Hàm helper (load data,...)
|   |-- knn.py              # Part A.1
|   |-- item_response.py    # Part A.2
|   |-- matrix_factorization.py # Part A.3 (Option i)
|   |-- neural_network.py   # Part A.3 (Option ii)
|   |-- ensemble.py         # Part A.4
|-- report/               # Chứa file báo cáo LaTeX
|   |-- final_report.tex
|   |-- llm_report.tex
|-- .gitignore
|-- README.md
`-- requirements.txt      # Danh sách thư viện
```

## 🚀 Hướng dẫn cài đặt và Khởi chạy

### Bước 1: Clone Repository

```bash
git clone [URL_CUA_REPO]
cd mla_final_project
```

### Bước 2: Tải dữ liệu

Tải các file dữ liệu (`train_data.csv`, `valid_data.csv`, `test_data.csv`, `sparse_matrix.npz`, `question_meta.csv`, `student_meta.csv`) và đặt chúng vào thư mục `data/`.

### Bước 3: Tạo môi trường ảo (Virtual Environment)

Việc sử dụng môi trường ảo là bắt buộc để thống nhất phiên bản thư viện.

```bash
# Tạo môi trường ảo (tên là 'venv')
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows (cmd):
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate
```
(Sau khi kích hoạt, `(venv)` sẽ xuất hiện ở đầu dòng lệnh).

### Bước 4: Cài đặt thư viện

Sử dụng file `requirements.txt` để cài đặt tất cả các thư viện cần thiết.

```bash
pip install -r requirements.txt
```

Các thư viện chính bao gồm:
* `numpy`
* `scipy`
* `pandas`
* `torch` (PyTorch)
* `scikit-learn` (cho confusion_matrix, roc_auc_score)
* `matplotlib` (để vẽ biểu đồ)

### Bước 5: Chạy thử Code

Để kiểm tra xem môi trường đã setup đúng chưa, hãy chạy file `utils.py` để test load data:

```bash
# Đảm bảo đang ở thư mục gốc (mla_final_project/)
# Chạy file utils.py từ trong thư mục src
python src/utils.py
```
Nếu output là "Tải dữ liệu thành công" và "Số dòng train:...", nghĩa là quá trình setup đã hoàn tất.

Để chạy thử một phần của dự án (ví dụ: `knn.py`), thực hiện lệnh:
```bash
python src/knn.py
```


## Kết quả knn
Shape của ma trận: (542, 1774)

==============================
Bắt đầu thử nghiệm User-based KNN...
==============================

[User-based, k=1]
Validation Accuracy: 0.6260
Confusion Matrix:
[[1338 1491]
 [1159 3098]]

[User-based, k=6]
Validation Accuracy: 0.6778
Confusion Matrix:
[[1033 1796]
 [ 487 3770]]

[User-based, k=11]
Validation Accuracy: 0.6895
Confusion Matrix:
[[1324 1505]
 [ 695 3562]]

[User-based, k=16]
Validation Accuracy: 0.6751
Confusion Matrix:
[[1088 1741]
 [ 561 3696]]

[User-based, k=21]
Validation Accuracy: 0.6681
Confusion Matrix:
[[1187 1642]
 [ 710 3547]]

[User-based, k=26]
Validation Accuracy: 0.6507
Confusion Matrix:
[[1033 1796]
 [ 679 3578]]

==============================
Bắt đầu thử nghiệm Item-based KNN...
==============================

[Item-based, k=1]
Validation Accuracy: 0.6121

[Item-based, k=6]
Validation Accuracy: 0.6606

[Item-based, k=11]
Validation Accuracy: 0.6798

[Item-based, k=16]
Validation Accuracy: 0.6873

[Item-based, k=21]
Validation Accuracy: 0.6919

[Item-based, k=26]
Validation Accuracy: 0.6909

Đã lưu dự đoán Item-based (k=21) vào file: 2201040051_item_knn_preds.npy

==============================
Báo cáo ROC-AUC (cho k tốt nhất trên tập Validation)
==============================
User-based (k=11) - ROC-AUC: 0.7362
Item-based (k=21) - ROC-AUC: 0.7392

==============================
Tổng kết và Báo cáo Test Accuracy
==============================
Mô hình tốt nhất: Item-based (k=21) với Val Acc = 0.6919
Đang chạy trên tập Test...

--- KẾT QUẢ CUỐI CÙNG (PART A.1) ---
[Summary] For K=21, mô hình KNN tốt nhất (Item-based) đạt được 0.6794 test accuracy.

Reflection (Yêu cầu 1d - Hãy chỉnh sửa lại bằng từ ngữ của riêng bạn):
Item-based KNN cho kết quả tốt nhất khi k=21. Phương pháp này hoạt động tốt hơn user-based, có thể vì việc 'giống nhau' giữa các câu hỏi là một chỉ báo mạnh hơn là sự 'giống nhau' giữa các học sinh.