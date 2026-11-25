import re
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import category_encoders as ce

def pipeline_housing(df2, Train, scaling_data=True, Handling_Outliers=True,
             Handling_Missing_Values_Not_DT=True, Handling_Missing_Values_DT=False,
             encoding=True, is_training_data=True):
    """
    df: dataframe to clean and prepare for modeling or deployment
    Train: dataframe used for training models (used to fit all transformers)

    Handling_Missing_Values_Not_DT: default True
        - Handling Missing Values for Non-Decision Tree Algorithms
        - For DT, Random Forest, Extra Trees, AdaBoost, XGBoost, LightGBM, CatBoost → set False
    Handling_Missing_Values_DT: default False
        - Handling Missing Values for Decision Tree Algorithms
        - For DT, Random Forest, Extra Trees, AdaBoost → set True

    is_training_data: default True
        - For test data or new application data → set False

    # For distance based algorithm and those rely on gradient descent based optimisation such as Linear Regression, Logistic Regression, SVM, Neural Networks:
    scaling_data=True,
    Handling_Outliers=True,
    Handling_Missing_Values_Not_DT=True,
    Handling_Missing_Values_DT=False,
    encoding=True

    # For DECISION TREES, RANDOM FOREST, and Extra Trees
    scaling_data=False,
    Handling_Outliers=False, #tree splits are not affected by outliers
    Handling_Missing_Values_Not_DT=False,
    Handling_Missing_Values_DT=True, ## DT, RF, Extra Trees do NOT handle NaN — sentinel fill (-9999999) used
    encoding=True

    # For ADABOOST
    scaling_data=False,
    Handling_Outliers=True, ## unlike RF/DT, AdaBoost is sensitive to outliers via sample reweighting
    Handling_Missing_Values_Not_DT=False,
    Handling_Missing_Values_DT=True,
    encoding=True

    # For XGBOOST and LIGHTGBM
    scaling_data=False,
    Handling_Outliers=False,
    Handling_Missing_Values_Not_DT=False,
    Handling_Missing_Values_DT=False,  # XGBOOST, LIGHTGBM, and CATBOOST handle missing values internally
    encoding=True

    # For CATBOOST
    scaling_data=False,
    Handling_Outliers=False,
    Handling_Missing_Values_Not_DT=False,
    Handling_Missing_Values_DT=False,  # XGBOOST, LIGHTGBM, and CATBOOST handle missing values internally
    encoding=False  # CATBOOST handle categorical internally
    """

    df=df2.copy()  # prevents unexpected side effects

    ##############################################################
    # 1. Duplicates & ID Columns
    ##############################################################

    #### Since property_id is just an id of the house with all unique values so drop this column
     # dropping features that are only for identification
    # safe drop — won't crash if columns are already absent (e.g. on test data)
    df.drop(columns=[c for c in ['property_id'] if c in df.columns], inplace=True)

    df.drop_duplicates(inplace=True)

    #'renovation_year' has 68% of the data missing so fill NaN with 0 as no information is available
    # 'lot_size_sqft' has 24% of missing data so fill with median

      ##############################################################
    # 2. Basic Cleaning
    ##############################################################

    df=df.fillna({'renovation_year': 0,
             'lot_size_sqft': df.lot_size_sqft.median()})

    

    ##############################################################
    # 3. Drop rows missing ALL values
    ##############################################################
    
    df.dropna(how='all', inplace=True)

    ##############################################################
    # 4. Missing Value Imputation
    ##############################################################
    if Handling_Missing_Values_Not_DT:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(exclude='number').columns.tolist()

        # Categorical: fill with mode from Train
        for col in cat_cols:
            if df[col].isna().any():
                df[col + "_missing"] = df[col].isna().astype(int)
            mode_val = Train[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

        # Numeric: MICE — fit on Train only to avoid leakage
        mice = IterativeImputer(max_iter=10, random_state=42, min_value=0)

        if is_training_data:
            mice.fit(Train[numeric_cols])
            joblib.dump(mice, 'mice_imputer.pkl')
        else:
            mice = joblib.load('mice_imputer.pkl')

        # create missing indicators column-by-column BEFORE imputation
        for col in numeric_cols:
            if df[col].isna().any():
                df[col + "_missing"] = df[col].isna().astype(int)
        df[numeric_cols] = mice.transform(df[numeric_cols])

    if Handling_Missing_Values_DT:
        for col in df.select_dtypes(include='number').columns:
            if df[col].isna().any():
                df[col + "_missing"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(9999999)

        for col in df.select_dtypes(exclude='number').columns:
            df[col] = df[col].fillna("Unknown")

    ##############################################################
    # 5. Outlier Handling (numeric columns only)
    ##############################################################
    if Handling_Outliers:
        # Iterate over df's numeric columns, excluding _missing indicators added in step 4
        # Outlier bounds (Q1, Q3, IQR) are always computed from Train to avoid leakage
        cols_to_check = [
            c for c in df.select_dtypes(include='number').columns
            if not c.endswith('_missing')
        ]

        for col in cols_to_check:
            Q1 = Train[col].quantile(0.25)
            Q3 = Train[col].quantile(0.75)
            IQR = Q3 - Q1
            LOF = Q1 - 3 * IQR  # Lower Outer Fence
            UOF = Q3 + 3 * IQR  # Upper Outer Fence
            df[col + "_flag"] = df[col].apply(lambda x: 1 if x < LOF or x > UOF else 0)
            df[col] = df[col].clip(lower=LOF, upper=UOF)

    
    ##############################################################
    # 6. Feature Engineering 
    ##############################################################

    ### 'renovation_year' has 3408 valyes as 0, representting no information about renovation/no renovation
    ### map it to renovated: 'yes:1'/'no:0'
    df['renovated']=df['renovation_year'].apply(lambda x: 1 if x >0 else 0)
    df=df.drop('renovation_year', axis=1)

    #Map 'garage': 0:No,1:Yes
    df['garage']=df['garage'].apply(lambda x: 1 if x=='Yes' else 0)

    # Map 'basement': No:0,Yes:1
    df['basement']=df['basement'].apply(lambda x: 1 if x=='Yes' else 0)

    PWest=['Gatineau','Rouyn-Noranda',"Val-d'Or",'Vaudreuil-Dorion','Dollard-des-Ormeaux','Pointe-Claire']
    PNorth=['Laval','Terrebonne','Repentigny','Blainville','Mirabel','Saint-Jérôme']
    PSouth=['Longueuil','Brossard','Boucherville','St-Bruno-de-Montarville','Candiac','Châteauguay','Saint-Jean-sur-Richelieu',\
           'Granby','Sherbrooke']
    PEast=['Québec','Lévis','Trois-Rivières','Drummondville','Victoriaville','Saguenay','Alma','Rimouski','Baie-Comeau','Sept-Îles']

    # Build the mapping dictionary
    city_map={}
    for city in PWest: city_map[city]='PWest'
    for city in PNorth: city_map[city]='PNorth'
    for city in PSouth: city_map[city]='PSouth'
    for city in PEast: city_map[city]='PEast'


    # Map dictionary to create new column
    df['ProvLocation']=df['city'].map(city_map)

    # transform() calculates the median for the group, but repeats (broadcasts) that value for every single     matching row.
    neighborhood_medians=df.groupby(['neighborhood','property_type'])['lot_size_sqft'].transform('median')
    df['sqft_relative_to_neighborhood']=round(df['lot_size_sqft']/neighborhood_medians,2)

    #  Get the neighborhood baseline year
    median_year=df.groupby(['city','neighborhood'])['year_built'].transform('median')
    #  Calculate how much newer or older the house is relative to its peers
    df['years_newer_than_neighborhood'] = df['year_built'] - median_year

    # if encoding:
    #     # Isolate the target column from your Train dataframe to fit the encoder
    #     y_train_internal=Train['sale_price']

    #     # Set up the encoder
    #     encoder = ce.TargetEncoder(cols=["city", "neighborhood"], smoothing=10.0)
        
    #     encoder.fit_transform(Train)
    #     encoder.transform(df)
     ##############################################################
    # 7. Scaling (numeric columns only)
    ##############################################################
    if scaling_data:
        # Exclude binary indicator/flag columns — scaling adds no benefit
        # and makes the values harder to interpret (0/1 becomes e.g. -0.5 / 2.3)
        numeric_cols = [
            c for c in df.select_dtypes(include='number').columns
            if not c.endswith('_missing') and not c.endswith('_flag')
        ]
        scaler = RobustScaler()  # RobustScaler uses Median & IQR — not sensitive to outliers

        if is_training_data:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            joblib.dump(scaler, 'scaler.pkl')
        else:
            scaler = joblib.load('scaler.pkl')
            df[numeric_cols] = scaler.transform(df[numeric_cols])

      ##############################################################
    # 8. Encoding (categorical columns only)
    ##############################################################
    if encoding:
        cat_cols = df.select_dtypes(exclude='number').columns.tolist()
        num_cols = df.select_dtypes(include='number').columns.tolist()

        # OHE is fitted on df's categorical columns after all prior cleaning steps
        # This is correct as long as is_training_data=True is called with the full training set
        # handle_unknown='ignore' silently zero-encodes unseen categories at inference —
        # intentional: new categories contribute nothing to the model output
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        if is_training_data:
            ohe_array = ohe.fit_transform(df[cat_cols])
            joblib.dump(ohe, 'ohe.pkl')
        else:
            ohe = joblib.load('ohe.pkl')
            ohe_array = ohe.transform(df[cat_cols])

        ohe_names = ohe.get_feature_names_out(cat_cols)
        df_cat = pd.DataFrame(ohe_array, columns=ohe_names, index=df.index)
        df = pd.concat([df[num_cols], df_cat], axis=1)

        ##############################################################
    # 9. Column Name Cleanup
    ##############################################################
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"[<]", "lt", regex=True)
        .str.replace(r"[>]", "gt", regex=True)
        .str.replace(r"[+]", "plus", regex=True)
        .str.replace(" ", "_")
    )
    df.columns = [re.sub(r'.*?__', '', c) for c in df.columns]

    return df


"""
##############################################################
# USAGE GUIDE
##############################################################

# Separate train and test BEFORE calling dataprep
# Always separate features and target BEFORE dataprep — never pass Loan Status into the pipeline

X_train = df_train.drop(columns=['Loan Status'])
y_train = df_train['Loan Status']

X_test = df_test.drop(columns=['Loan Status'])
y_test = df_test['Loan Status']


# ── Logistic Regression / SVM / Neural Network ──────────────
X_train_clean = dataprep(
    df2=X_train,
    Train=X_train,         # Train = itself during training
    scaling_data=True,
    Handling_Outliers=True,
    Handling_Missing_Values_Not_DT=True,
    Handling_Missing_Values_DT=False,
    encoding=True,
    is_training_data=True
)
y_train_scaled = y_train.loc[X_train_scale.index]
model.fit(X_train_clean, y_train)
joblib.dump(model, 'model.pkl')

X_test_clean = dataprep(
    df2=X_test,
    Train=X_train,         # always pass the original training data — never refit on test
    scaling_data=True,
    Handling_Outliers=True,
    Handling_Missing_Values_Not_DT=True,
    Handling_Missing_Values_DT=False,
    encoding=True,
    is_training_data=False
)
model = joblib.load('model.pkl')

## If you used pd.get_dummies() instead of OHE — then yes, reindex is essential 
X_test_clean = X_test_clean.reindex(columns=X_train_clean.columns, fill_value=0)
predictions = model.predict(X_test_clean)


# ── Decision Trees / Random Forest / Extra Trees ─────────────
X_train_clean = dataprep(X_train, X_train,
    scaling_data=False, Handling_Outliers=False,
    Handling_Missing_Values_Not_DT=False, Handling_Missing_Values_DT=True,
    encoding=True, is_training_data=True)


# ── AdaBoost ─────────────────────────────────────────────────
X_train_clean = dataprep(X_train, X_train,
    scaling_data=False, Handling_Outliers=True,
    Handling_Missing_Values_Not_DT=False, Handling_Missing_Values_DT=True,
    encoding=True, is_training_data=True)


# ── XGBoost / LightGBM ───────────────────────────────────────
X_train_clean = dataprep(X_train, X_train,
    scaling_data=False, Handling_Outliers=False,
    Handling_Missing_Values_Not_DT=False, Handling_Missing_Values_DT=False,
    encoding=True, is_training_data=True)


# ── CatBoost ─────────────────────────────────────────────────
X_train_clean = dataprep(X_train, X_train,
    scaling_data=False, Handling_Outliers=False,
    Handling_Missing_Values_Not_DT=False, Handling_Missing_Values_DT=False,
    encoding=False,        # CatBoost handles categorical internally
    is_training_data=True)
"""