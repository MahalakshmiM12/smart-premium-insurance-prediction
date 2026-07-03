# SmartPremium — Insurance Premium Prediction System

SmartPremium is an end-to-end machine learning project that predicts insurance premium amounts based on customer demographic, financial, and policy information. The project covers the full ML lifecycle — exploratory data analysis, preprocessing, model training with experiment tracking, batch prediction, and an interactive web app for real-time predictions.


---

## Features

- **Exploratory Data Analysis (EDA)** — distribution plots, missing value analysis, outlier detection, and correlation analysis
- **Preprocessing Pipeline** — reusable `ColumnTransformer` with median/mode imputation, scaling, and one-hot encoding
- **Model Training** — trains and compares Linear Regression, Decision Tree, Random Forest, and XGBoost regressors with `RandomizedSearchCV` for hyperparameter tuning
- **Experiment Tracking** — all runs, metrics, and models logged with **MLflow**
- **Batch Prediction** — generates submission file from the test set
- **Streamlit Web App** — interactive UI for predicting premiums from user-entered customer details

---

## Business Value

- Optimize premium pricing based on risk factors
- Assess risk for loan approvals tied to insurance policies
- Estimate future healthcare costs for patients
- Provide real-time insurance quotes through data-driven predictions
---

## Pipeline Overview

```
eda.ipynb  →  preprocessing.ipynb  →  train_model.ipynb  →  predict.ipynb / smart_premium_app.py
```

1. **`eda.ipynb`** — Loads the raw train/test data and explores it: shape, dtypes, missing values, duplicates, numerical vs. categorical breakdown, target distribution, univariate/bivariate analysis, outlier detection, correlation heatmap, and `Policy Start Date` trends.

2. **`preprocessing.ipynb`** — Cleans the data and builds a reusable preprocessing pipeline:
   - Drops `id` and `Customer Feedback`
   - Extracts `Policy_Year`, `Policy_Month`, `Policy_Day` from `Policy Start Date`, then drops the original date column
   - Numerical features → median imputation + `StandardScaler`
   - Categorical features → most-frequent imputation + `OneHotEncoder`
   - Combines both via `ColumnTransformer`
   - Saves the fitted preprocessor to `models/preprocessor.pkl`

3. **`train_model.ipynb`** — Loads the saved preprocessor, wraps it in a `Pipeline` with each candidate model, tunes hyperparameters with `RandomizedSearchCV`, evaluates on a held-out test split (RMSE, MAE, R²), logs every run to **MLflow**, and saves the best-performing model to `models/best_model.pkl`.

4. **`predict.ipynb`** — Loads `best_model.pkl`, applies the same feature engineering to the Kaggle test set, generates predictions, and writes `data/submission.csv`.

5. **`smart_premium_app.py`** — A Streamlit app where a user can input customer details through a form and get an instant predicted premium from the trained model.

---

## How to Run this Project

**1. Clone the repository**
```bash
git clone https://github.com/MahalakshmiM12/smart-premium-insurance-prediction.git
cd SmartPremium
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add the dataset**

Download dataset and place the files here:
```
data/playground-series-s4e12/train.csv
data/playground-series-s4e12/test.csv
data/playground-series-s4e12/sample_submission.csv
```

---

## Usage

Run the notebooks in order (Jupyter Notebook / JupyterLab), from the project root so relative paths resolve correctly:

```bash
jupyter notebook
```

1. Run **`eda.ipynb`** to explore the data
2. Run **`preprocessing.ipynb`** to generate `models/preprocessor.pkl`
3. Run **`train_model.ipynb`** to train, tune, track, and save `models/best_model.pkl`
4. (Optional) Run **`predict.ipynb`** to generate `data/submission.csv` for the test set

### Track experiments with MLflow

Model runs are logged to a local SQLite backend. To view them:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open [http://localhost:5000](http://localhost:5000) in your browser.

### Launch the web app

```bash
streamlit run smart_premium_app.py
```
Enter customer details in the form and click **Predict Premium** to get an instant estimate.

---

## Tech Stack

- **Language:** Python
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn, XGBoost
- **Experiment Tracking:** MLflow (SQLite backend)
- **Model Persistence:** joblib
- **Web App:** Streamlit

---

## Notes

- All file paths use `os.path.join()` with a project-root-relative base for cross-platform portability — run notebooks from the project root.
- The same feature engineering (dropping `id`/`Customer Feedback`, extracting `Policy_Year`/`Policy_Month`/`Policy_Day`) must be applied identically wherever raw data is loaded (training, prediction, and the Streamlit app) to match what the model was trained on.

---

## License

This project is open source and available under the [MIT License](LICENSE).
