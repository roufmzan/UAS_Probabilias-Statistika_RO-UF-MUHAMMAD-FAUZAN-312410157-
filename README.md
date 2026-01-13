# Loan Eligibility / Loan Approval Prediction (Machine Learning)

Proyek ini memprediksi **status persetujuan pinjaman** (kolom `Loan_Status`) berdasarkan informasi pemohon seperti gender, status menikah, pendapatan, dan jumlah pinjaman. Implementasi utama ada di notebook `Loan_Eligibility_Prediction.ipynb`.

## Isi Repository

- `Loan_Eligibility_Prediction.ipynb`
- `loan_dataset.csv`

## Dataset

Dataset berada di `loan_dataset.csv` dengan kolom:

- `Loan_ID` (ID unik)
- `Gender` (kategorikal)
- `Married` (kategorikal)
- `ApplicantIncome` (numerik)
- `LoanAmount` (numerik, ada nilai kosong/NaN)
- `Loan_Status` (target: `Y`/`N`)

Ukuran dataset pada notebook: **598 baris, 6 kolom**.

## Alur di Notebook

1. **Load data**: membaca `loan_dataset.csv`.
2. **EDA**:
   - pie chart distribusi `Loan_Status`
   - countplot `Gender` dan `Married` terhadap `Loan_Status`
   - distribusi dan boxplot untuk `ApplicantIncome` dan `LoanAmount`
3. **Outlier handling**:
   - filter `ApplicantIncome < 25000`
   - filter `LoanAmount < 400000`
4. **Encoding**:
   - kolom bertipe `object` di-*label encode* memakai `LabelEncoder`.
5. **Split data**:
   - `train_test_split(test_size=0.2, random_state=10)`
6. **Imbalanced handling**:
   - oversampling kelas minoritas dengan `RandomOverSampler`.
7. **Scaling**:
   - normalisasi fitur dengan `StandardScaler`.
8. **Model**:
   - `SVC(kernel='rbf')`
9. **Evaluasi**:
   - ROC AUC (train/validasi)
   - confusion matrix
   - classification report

Catatan: hasil metrik dapat sedikit berbeda tergantung versi library/lingkungan eksekusi.

## Menjalankan Project

### 1) Buat virtual environment (opsional tapi disarankan)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependensi

Jika kamu belum punya dependensi yang diperlukan, install ini:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### 3) Jalankan Jupyter Notebook

```bash
jupyter notebook
```

Lalu buka file `Loan_Eligibility_Prediction.ipynb` dan jalankan cell dari atas ke bawah.

## Output yang Dihasilkan

Notebook menghasilkan:

- visualisasi EDA (pie chart, countplot, distplot, boxplot)
- heatmap korelasi
- nilai ROC AUC train dan validasi
- confusion matrix
- classification report

## Catatan Teknis

- Encoding dilakukan ke **semua** kolom bertipe `object` (termasuk `Loan_ID`).
- Oversampling hanya diterapkan pada data train.
- Evaluasi ROC AUC pada notebook dihitung menggunakan `roc_auc_score(y_true, model.predict(X))` (menggunakan label prediksi, bukan probabilitas).
