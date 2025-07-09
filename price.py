import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
import joblib
import lightgbm as lgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

# Suppress TensorFlow logs
tf.get_logger().setLevel('ERROR')
# Suppress TensorFlow warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("combined_df.csv")
ZONE_KEYWORDS = {
    'dwarka': 'Dwarka',
    'rohini': 'Rohini',
    'karol bagh': 'Karol Bagh / Central',
    'central': 'Karol Bagh / Central',
    'greater noida west': 'Greater Noida West',
    'noida extension': 'Greater Noida West',
    'sector 150': 'Noida Sec 150 / Golf City',
    'golf city': 'Noida Sec 150 / Golf City',
    'east delhi': 'East Delhi',
    'gurugram': 'Gurugram (DLF, Sohna Rd)',
    'sohna': 'Gurugram (DLF, Sohna Rd)',
    'indirapuram': 'Indirapuram',
    'raj nagar extension': 'Raj Nagar Extension',
    'vaishali': 'Vaishali',
    'vasundhara': 'Vasundhara',
    'lajpat nagar': 'Lajpat Nagar',
    'saket': 'Saket',
    'connaught place': 'Connaught Place',
    'pitampura': 'Pitampura',
    'punjabi bagh': 'Punjabi Bagh',
    'paschim vihar': 'Paschim Vihar',
    'mayur vihar': 'Mayur Vihar',
    'dwarka mor': 'Dwarka Mor',
    'shahdara': 'Shahdara',
    'sector 62': 'Noida Sector 62',
    'sector 137': 'Noida Sector 137',
}

ZONE_MULTIPLIERS = {
    'Dwarka': 1.20,
    'Rohini': 1.10,
    'Karol Bagh / Central': 1.25,
    'Greater Noida West': 1.30,
    'Noida Sec 150 / Golf City': 1.35,
    'East Delhi': 1.05,
    'Gurugram (DLF, Sohna Rd)': 1.40,
    'Indirapuram': 1.20,
    'Raj Nagar Extension': 1.15,
    'Vaishali': 1.15,
    'Vasundhara': 1.15,
    'Lajpat Nagar': 1.25,
    'Saket': 1.25,
    'Connaught Place': 1.30,
    'Pitampura': 1.15,
    'Punjabi Bagh': 1.15,
    'Paschim Vihar': 1.15,
    'Mayur Vihar': 1.15,
    'Dwarka Mor': 1.15,
    'Shahdara': 1.10,
    'Noida Sector 62': 1.25,
    'Noida Sector 137': 1.25,
    'Other NCR': 1.00,
}


def map_zone(area_name):
    if pd.isna(area_name):
        return "Other NCR"
    name_clean = str(area_name).lower().strip()
    for keyword, zone in ZONE_KEYWORDS.items():
        if keyword in name_clean:
            return zone
    return "Other NCR"


def get_zone_multiplier(zone):
    return ZONE_MULTIPLIERS.get(zone, 1.10)


def get_property_multiplier(bhk):
    bhk = int(bhk)
    if bhk == 1:
        return 1.08
    elif bhk == 2:
        return 1.15
    elif bhk == 3:
        return 1.10
    else:
        return 1.05

YEAR_MULTIPLIER = 1

# Helper for finding average price
def find_best_match_area(user_area_name, lookup_df, value_col, default_value):
    user_area_name = user_area_name.lower().strip()

    exact_match = lookup_df.loc[
        lookup_df["area_name"] == user_area_name
    ]
    if len(exact_match) > 0:
        return exact_match[value_col].values[0]

    partial_matches = lookup_df[
        lookup_df["area_name"].apply(
            lambda x: user_area_name in x or x in user_area_name
        )
    ]
    if len(partial_matches) > 0:
        return partial_matches.iloc[0][value_col]

    return default_value

# 1. CLEANING + PREPROCESSING FUNCTIONS
def preprocess_area(text: str) -> str:
    """
    Clean area name text.
    """
    text = text.replace("delhi ncr", "")
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    text = text.lower().strip()
    return re.sub(r'\s+', ' ', text)


def prepare_tfidf(train_texts, test_texts, max_features=50):
    """
    Fit TF-IDF on training area names and transform both train/test.
    """
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(train_texts).toarray()
    X_test_tfidf = tfidf.transform(test_texts).toarray()

    feature_names = tfidf.get_feature_names_out()
    return X_train_tfidf, X_test_tfidf, feature_names, tfidf


def one_hot_encode(train_df, test_df, categorical_cols):
    """
    One-hot encode categorical columns consistently for train and test.
    """
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    ohe_train = ohe.fit_transform(train_df[categorical_cols])
    ohe_test = ohe.transform(test_df[categorical_cols])
    
    col_names = ohe.get_feature_names_out(categorical_cols)
    
    train_ohe_df = pd.DataFrame(ohe_train, columns=col_names, index=train_df.index)
    test_ohe_df = pd.DataFrame(ohe_test, columns=col_names, index=test_df.index)
    
    return train_ohe_df, test_ohe_df, ohe


# 2. MODEL BUILDING
def build_dnn_model(input_dim, learning_rate=0.0005):
    """
    Create a compiled Keras DNN model for regression.
    """
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model


def evaluate_model(y_true, y_pred, label="Model"):
    """
    Print RMSE, MAE, and R² scores.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)    
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{label} R²: {r2:.4f}")
    print(f"{label} RMSE: ₹{rmse:,.0f}")
    print(f"{label} MAE: ₹{mae:,.0f}")
    return rmse, mae, r2


# 3. PIPELINE FUNCTION
## This function is the entire pipeline from data preparation to model training and evaluation to  be run once to save the models and encoders.

def run_pipeline(df):

    df = df.copy()

    # Clean area names
    df['area_name'] = df['area_name'].apply(preprocess_area)
    avg_price_lookup = (
        df.groupby("area_name")["avg_area_price"]
        .mean()
        .reset_index()
    )

    # Split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # TF-IDF
    X_train_tfidf, X_test_tfidf, tfidf_cols, tfidf_vectorizer = prepare_tfidf(
        df_train['area_name'],
        df_test['area_name'],
        max_features=50
    )

    tfidf_train_df = pd.DataFrame(X_train_tfidf, columns=tfidf_cols, index=df_train.index)
    tfidf_test_df = pd.DataFrame(X_test_tfidf, columns=tfidf_cols, index=df_test.index)

    # Add TF-IDF to train/test
    df_train_enc = pd.concat([df_train.reset_index(drop=True), tfidf_train_df.reset_index(drop=True)], axis=1)
    df_test_enc = pd.concat([df_test.reset_index(drop=True), tfidf_test_df.reset_index(drop=True)], axis=1)

    # One-hot encoding
    cat_cols = ['furnishing','status','transaction','type']
    ohe_train, ohe_test, ohe = one_hot_encode(df_train_enc, df_test_enc, cat_cols)

    df_train_enc = pd.concat([df_train_enc, ohe_train], axis=1)
    df_test_enc = pd.concat([df_test_enc, ohe_test], axis=1)

    # Drop unnecessary columns
    drop_cols = ['area_name','furnishing','status','transaction','type',
                 'price_per_bhk','per_sqft','log_price','price','price_per_sqft_per_bhk']

    X_train = df_train_enc.drop(columns=drop_cols)
    y_train = df_train_enc['price'].values

    X_test = df_test_enc.drop(columns=drop_cols)
    y_test = df_test_enc['price'].values

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    y_pred_rf = rf.predict(X_test_scaled)
    evaluate_model(y_test, y_pred_rf, label="Random Forest")

    # DNN
    dnn = build_dnn_model(X_train_scaled.shape[1])
    dnn.fit(X_train_scaled, y_train,
            validation_split=0.2,
            epochs=200,
            batch_size=32,
            verbose=0)

    y_pred_dnn = dnn.predict(X_test_scaled).flatten()
    evaluate_model(y_test, y_pred_dnn, label="DNN")


    #lgb predictions
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=10,
        num_leaves=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    lgb_model.fit(X_train_scaled, y_train)
    y_pred_lgb = lgb_model.predict(X_test_scaled)
    evaluate_model(y_test, y_pred_lgb, label="LightGBM")

    # Stacking model
    estimators = [
        ('rf', rf),
        ('lgb', lgb_model),
    ]

    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )

    stack.fit(X_train_scaled, y_train)
    y_pred_stack = stack.predict(X_test_scaled)
    evaluate_model(y_test, y_pred_stack, label="Stacking Ensemble")


    stacked_preds_train = np.vstack([
        y_pred_dnn,
        y_pred_rf,
        y_pred_lgb,
        y_pred_stack
    ]).T

    # Fit ensemble regressor
    meta_ensemble = Ridge(alpha=1.0)
    meta_ensemble.fit(stacked_preds_train, y_test)

    # Predict ensemble
    y_pred_meta = meta_ensemble.predict(stacked_preds_train)
    evaluate_model(y_test, y_pred_meta, label="Meta Ensemble")

    # Save random forest
    joblib.dump(rf, "model/rf_model.pkl")
    # Save scaler
    joblib.dump(scaler, "model/scaler.pkl")
    # Save TF-IDF vectorizer
    joblib.dump(tfidf_vectorizer, "model/tfidf.pkl")
    # Save OHE encoder
    joblib.dump(ohe, "model/ohe_encoder.pkl")
    # Save DNN separately
    dnn.save("model/dnn_model.keras")
    # Save lgb model
    joblib.dump(lgb_model, "model/lgb_model.pkl")
    # Save stacking model
    joblib.dump(stack, "model/stack_model.pkl")
    # Save meta ensemble model
    joblib.dump(meta_ensemble, "model/meta_ensemble.pkl")
    # Save area lookup
    joblib.dump(avg_price_lookup, "model/avg_price_lookup.pkl")
    # Save feature list
    joblib.dump(X_train.columns.tolist(), "model/feature_names.pkl")



    return {
        "rf_model": rf,
        "dnn_model": dnn,
        "lgb_model": lgb_model,
        "stack_model": stack,
        "scaler": scaler,
        "tfidf_vectorizer": tfidf_vectorizer,
        "ohe_encoder": ohe,
        "avg_price_lookup": avg_price_lookup,
        "meta_ensemble": meta_ensemble,
        "feature_names": X_train.columns.tolist(),
    }


# 4. OUTFLOW PREDICTION FUNCTION
def predict_price(new_data, pipeline_dict, model_type='dnn'):
    """
    Predict price for new data rows using trained pipeline.
    
    Parameters:
        new_data (DataFrame): new rows to predict
        pipeline_dict (dict): returned from run_pipeline()
    
    Returns:
        array: predicted prices
    """
    # Clean area_name
    new_data = new_data.copy()
    new_data['area_name'] = new_data['area_name'].apply(preprocess_area)
    
    # TF-IDF
    tfidf_vectorizer = pipeline_dict['tfidf_vectorizer']
    X_tfidf = tfidf_vectorizer.transform(new_data['area_name']).toarray()
    tfidf_df = pd.DataFrame(X_tfidf, columns=tfidf_vectorizer.get_feature_names_out())

    # Merge TF-IDF into new_data
    new_data_enc = pd.concat([new_data.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    
    # One-hot encoding
    cat_cols = ['furnishing','status','transaction','type']
    ohe = pipeline_dict["ohe_encoder"]
    
    ohe_array = ohe.transform(new_data_enc[cat_cols])
    ohe_df = pd.DataFrame(
        ohe_array,
        columns=ohe.get_feature_names_out(cat_cols),
        index=new_data_enc.index
    )
    
    # Add to new_data
    new_data_enc = pd.concat([new_data_enc, ohe_df], axis=1)
    
    # Drop same columns
    drop_cols = ['area_name','furnishing','status','transaction','type',
                 'price_per_bhk','per_sqft','log_price','price','price_per_sqft_per_bhk']
    
    for col in drop_cols:
        if col in new_data_enc.columns:
            new_data_enc = new_data_enc.drop(columns=[col])
    
    # Add any missing columns
    final_features = pipeline_dict['feature_names']
    for col in final_features:
        if col not in new_data_enc.columns:
            new_data_enc[col] = 0.0
    
    # Reorder columns
    new_data_enc = new_data_enc[final_features]
    
    # Scale
    scaler = pipeline_dict['scaler']
    X_scaled = scaler.transform(new_data_enc)
    # Load ensemble
    
    # Predict
    preds_dnn = pipeline_dict['dnn_model'].predict(X_scaled).flatten()
    preds_rf = pipeline_dict['rf_model'].predict(X_scaled).flatten()
    preds_lgb = pipeline_dict['lgb_model'].predict(X_scaled).flatten()
    preds_stack = pipeline_dict['stack_model'].predict(X_scaled).flatten()
    preds_ens = pipeline_dict['meta_ensemble']

    stacked_preds = np.vstack([
        preds_dnn,
        preds_rf,
        preds_lgb,
        preds_stack
    ]).T

    preds = preds_ens.predict(stacked_preds)

    preds_adj = (0.5 * preds_dnn + 0.2 * preds_lgb + 0.3 * preds_rf)

    # Compute multipliers
    new_data["zone"] = new_data["area_name"].apply(map_zone)
    new_data["zone_multiplier"] = new_data["zone"].apply(get_zone_multiplier)
    new_data["property_multiplier"] = new_data["bhk"].apply(get_property_multiplier)
    new_data["year_multiplier"] = YEAR_MULTIPLIER
    new_data["total_multiplier"] = (
        new_data["zone_multiplier"]
        * new_data["property_multiplier"]
        * new_data["year_multiplier"]
    )

    preds_adjusted_dnn = preds_dnn * new_data["total_multiplier"].values
    preds_adjusted_lgb = preds_lgb * new_data["total_multiplier"].values
    preds_adjusted_rf = preds_rf * new_data["total_multiplier"].values
    preds_adjusted_ens = preds * new_data["total_multiplier"].values
    preds_adjusted_adj = preds_adj * new_data["total_multiplier"].values
    return preds_adjusted_dnn, preds_adjusted_lgb, preds_adjusted_rf, preds_adjusted_ens, preds_adjusted_adj
    
#pipeline_dict = run_pipeline(df)

