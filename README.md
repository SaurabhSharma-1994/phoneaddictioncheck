# 📱 PhoneCheck — Smartphone Addiction Analysis

A machine learning dashboard to predict smartphone addiction using a **Random Forest classifier** trained on 7,500 users.

**Model Accuracy: 93.67%**

---

## 🚀 Deploy on Streamlit Cloud (Free, 5 minutes)

### Step 1 — Push to GitHub

```bash
# Create a new GitHub repo at github.com/new (name: phonecheck-app)
# Then run:

git init
git add .
git commit -m "Initial commit — PhoneCheck app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/phonecheck-app.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo `phonecheck-app`
5. Set **Main file path** → `app.py`
6. Click **Deploy!**

Your app will be live at:  
`https://YOUR_USERNAME-phonecheck-app-app-XXXX.streamlit.app`

---

## 💻 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
phonecheck-app/
├── app.py                          ← Streamlit app (main file)
├── model_rf.pkl                    ← Trained Random Forest model
├── scaler.pkl                      ← StandardScaler (fitted on train set)
├── good_features.json              ← Selected feature names
├── Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv  ← Dataset
├── requirements.txt                ← Python dependencies
└── README.md                       ← This file
```

---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Estimators | 100 trees |
| Max Depth | 10 |
| Train Size | 6,000 (80%) |
| Test Size | 1,500 (20%) |
| **Accuracy** | **93.67%** |
| Target | `addicted_label` (0 = Not Addicted, 1 = Addicted) |

### Features Used
- `daily_screen_time_hours`
- `social_media_hours`
- `sleep_hours`
- `weekend_screen_time`
- `screen_to_sleep_ratio` *(engineered)*
- `fun_usage_percent` *(engineered)*

---

## 📊 Dataset

- **Source:** [Kaggle — Smartphone Usage and Addiction Analysis](https://www.kaggle.com/datasets/algozee/smartphone-usage-and-addiction-analysis-dataset)
- **Records:** 7,500
- **Features:** 16 columns
- **Age Range:** 18–35
- **Addiction Rate:** 70.8%

---

## ⚠️ Data Leakage Note

The `addiction_level` column was intentionally dropped before training. It is derived from the target variable and would cause artificially inflated (~100%) accuracy via data leakage.
