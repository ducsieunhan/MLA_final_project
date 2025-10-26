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
