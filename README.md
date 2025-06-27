# ✈️ Flight Price Feature Engineering for Predictive Modeling

This project focuses on preparing a real-world flight pricing dataset for machine learning using extensive **feature engineering** and **data preprocessing** techniques. The goal is to create a clean, fully numeric dataset suitable for predictive modeling (e.g., price prediction).

---

## 📌 Project Objectives

- Clean and preprocess flight booking data
- Extract meaningful time-based features (e.g., departure/arrival hours)
- Handle missing and inconsistent values
- Transform textual durations into structured numerical values
- Apply One-Hot Encoding on categorical features
- Prepare a machine-learning-ready dataset

---

## 🛠️ Tools & Technologies

- **Python** 🐍
- **Pandas** & **NumPy** for data wrangling
- **Matplotlib** & **Seaborn** for EDA
- **Scikit-learn** for encoding and preprocessing

---

## 🔍 Key Features Engineered

| Feature | Description |
|--------|-------------|
| `Date`, `Month`, `Year` | Extracted from `Date_of_Journey` |
| `Dep_hour`, `Dep_min` | Extracted from `Dep_Time` |
| `Arrival_hour`, `Arrival_min` | Extracted from `Arrival_Time` |
| `Duration_hour`, `Duration_min` | Split and cleaned from `Duration` |
| `Total_Stops` | Converted to numeric labels |
| `Airline`, `Source`, `Destination` | One-hot encoded |

---

## 🧹 Data Cleaning & Handling

- Converted object datatypes to `int` or `float`
- Removed inconsistent entries like `'5m'`
- Replaced `<NA>`/`NaN` values with column medians
- Dropped redundant columns like `Route`, `Duration`, `Date_of_Journey`

---

## 🔄 One-Hot Encoding

Used `OneHotEncoder` from `sklearn` to transform multi-category columns:
```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df[['Airline', 'Source', 'Destination']])
encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out())
