




# 🏠 Quebec Housing Price Prediction


An end-to-end machine learning project predicting residential property prices across Quebec (including Montreal, Quebec City, Laval, and Gatineau) using tabular housing data from Kaggle.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology & Pipeline](#-methodology--pipeline)
- [Model Performance](#-model-performance)
- [Future Improvements](#-future-improvements)

---

## 🔍 Overview
Real estate valuation in Quebec varies significantly based on location, property type (condo, single-family detached, triplex), square footage, and proximity to transit or urban cores. The goal of this project is to build a regression model that accurately estimates property listing/sale prices and identifies the strongest price drivers.

---

## 📊 Dataset
The dataset is sourced from Kaggle and contains residential listings across Quebec.
https://www.kaggle.com/datasets/rickyjackson/quebec-housing-sales-price


- **Key Features:**
  - `city` / `neighborhood` / `postal_code`: Geographic location indicators
  - `property_type`: Single-family home, Condo, Duplex/Triplex
  - `bedrooms` & `bathrooms`: Room counts
  - `living_area_sqft` & `lot_size_sqft`: Interior and land dimensions
  - `year_built`: Construction year
 
- **Target Variable:** `price` (CAD)


---

## 📁 Project Structure
```text
quebec-housing-price-prediction/
├── data/
│   ├── raw/                  # Original Kaggle CSV files
│   └── processed/            # Cleaned & feature-engineered datasets
├── notebooks/
│   ├── Capstone Housing Notebook         # Exploratory Data Analysis & visualizations
│   └── modeling notebook     # Model training, tuning & SHAP interpretation
|   └── Pipeline Notebook     # Pipeline for EDA, Encoding and Scaling Model appropriate

└── README.md
```

---

## ⚙️ Methodology & Pipeline

1. **Exploratory Data Analysis (EDA):**
   - Check the duplicates in the data
   - Conducted Univariate and Bivariate analysis using statistical tests like Correlation, ANOVA and Chi-Square tests
  
3. **Data Preprocessing:**
   - Handled missing values in renovation_year by mapping it to a new column 'renovated' : 0 and 1
   - Missing values in 'lot_size_sqft ' filled with median
   - Encoded high-cardinality location features using One hot Encoding 
4. **Feature Engineering:**
   - Created a new feature 'ProvLocation' by mapping cities to the location in the province as East, West, North, South
   - 'sqft_relative_to_neighborhood' is created showing how large the property is relative to its neighborhood
   - 'years_newer_than_neighborhood' is created showing how much newer or older the house is relative to its peers

5. **Modeling & Validation:**
   - Models like XGBoost, CatBoost, LightGBM, Linear Regression are evaluated .
   - Hyperparameter tuning performed with GridSearchCV.

---

## 📈 Model Performance

| Model | RMSE (CAD) | Train-Test Gap | $R^2$ Score |
| :--- | :---: | :---: | :---: |


| **LightGBM Regressor (Tuned)** | **$53362.45** | **-0.0271** | **0.9058** |
| XGBoost Regressor | $56285.162 | -0.0752 | 0.8952 |



---
<img width="770" height="934" alt="download (1)" src="https://github.com/user-attachments/assets/7e7bc007-ea2d-4da6-902b-12e59d7f3296" />

SHAP Summary shows living_area_sqft:
"Living Area (SqFt)" at the top with:
- Red dots (large houses) on the RIGHT → High square footage INCREASES the price.
- Blue dots (small houses) on the LEFT → Low square footage DECREASES the price.


---

## 🔮 Future Improvements
- [ ] Integrate geospatial features (distance to nearest REM/Metro station, schools, and parks).
- [ ] Incorporate Bank of Canada interest rate trends if listing dates span multiple years.
- [ ] Deploy the best model as an interactive web app 






## Contact
- **Sobia Pervaiz** — [LinkedIn Profile]https://www.linkedin.com/in/sobia-pervaiz-071848215/ | 
