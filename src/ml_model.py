import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.ensemble import HistGradientBoostingRegressor
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ed_hospital_modeling.csv"
)
df = pd.read_csv(
    PROCESSED_DATA_PATH,
    dtype={
        "Facility ID": "string"
    }
)


TARGET = "OP_18a"
FEATURES = [
    "ed_volume",
    "Hospital Type",
    "Hospital Ownership",
    "State"
]
model_df = df[
    [
        "Facility ID",
        "Facility Name",
        "State",
        "OP_18a",
        "ed_volume",
        "Hospital Type",
        "Hospital Ownership"
    ]
].dropna(
    subset=[
        TARGET,
        *FEATURES
    ]
).copy()

print("ML sample shape:", model_df.shape)

X = model_df[FEATURES].copy()
y = model_df[TARGET].copy()

#print("X shape:", X.shape)
#print("Y shape:", Y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#print("\nTrain size:", len(X_train))
#print("\nTest size:", len(X_test))

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
dummy_mae = mean_absolute_error(y_test, dummy_pred)
dummy_rsme = mean_squared_error(y_test, dummy_pred)**0.5
dummy_r2 = r2_score(y_test, dummy_pred)
print("\nDummy Baseline")
print(f"MAE: {dummy_mae:.3f}")
print(f"R2: {dummy_r2:.3f}")
print(f"RSME: {dummy_rsme:.3f}")

"""Ridge Regression Model"""

categorical_features = FEATURES

preprocessor = ColumnTransformer(
    transformers=[
        (
            "catecorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)
ridge = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=1))
    ]
)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
ridge_rsme = mean_squared_error(y_test, ridge_pred)**0.5
print("\nRidge Regression Model")
print(f"MAE: {ridge_mae:.3f}")
print(f"R2: {ridge_r2:.3f}")
print(f"RSME: {ridge_rsme:.3f}")

param_grid = {
    "model__alpha": [0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
}

ridge_search = GridSearchCV(
    estimator=ridge,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1,
)

ridge_search.fit(X_train, y_train)
print("\nBest Ridge alpha:")
print(ridge_search.best_params_)
print("\nBest CV MAE:")
print(-ridge_search.best_score_)
#Optimizing MAE is more operational in this case for E[Y - Y']

best_ridge = ridge_search.best_estimator_
tuned_ridge_pred = best_ridge.predict(X_test)
tuned_ridge_mae = mean_absolute_error(
    y_test, tuned_ridge_pred
)
tuned_ridge_rmse = mean_squared_error(y_test, tuned_ridge_pred)**0.5
tuned_ridge_r2 = r2_score(y_test, tuned_ridge_pred)

print("\nTuned Ridge Test:")
print(f"MAE: {tuned_ridge_mae:.3f}")
print(f"R2: {tuned_ridge_r2:.3f}")
print(f"RMSE: {tuned_ridge_rmse:.3f}")

"""Random Forest Model"""
rf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
        )
    ]
)

rf.fit(X_train, y_train)
rf_train_pred = rf.predict(X_train)

rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
rf_train_rmse = mean_squared_error(y_train, rf_train_pred)**0.5
rf_train_r2 = r2_score(y_train, rf_train_pred)
print("\nRandom Forest Training:")
print(f"MAE: {rf_train_mae:.3f}")
print(f"R2: {rf_train_r2:.3f}")
print(f"RMSE: {rf_train_rmse:.3f}")

rf_cv = cross_validate(
    rf,
    X_train,
    y_train,
    cv=5,
    scoring={
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2"
    },
    n_jobs=-1,
)

rf_cv_mae = -rf_cv["test_mae"].mean()
rf_cv_rmse = -rf_cv["test_rmse"].mean()
rf_cv_r2 = rf_cv["test_r2"].mean()
print("\nRandom Forest 5-Fold CV:")
print(f"MAE: {rf_cv_mae:.3f}")
print(f"R2: {rf_cv_r2:.3f}")
print(f"RMSE: {rf_cv_rmse:.3f}")
#insuffient evidence than the untuned rf is better than ridge
rf_param_grid = {
    "model__max_depth": [None, 5, 10, 20],
    "model__min_samples_leaf": [2, 5, 10, 20],
    "model__max_features": ["sqrt", 0.5, 1.0],
}
rf_search = GridSearchCV(
    estimator=rf,
    param_grid=rf_param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1,
)

rf_search.fit(X_train, y_train)
print("\nBest Random Forest Params:")
print(rf_search.best_params_)
print(
    f"Best RF CV MAE: {-rf_search.best_score_:.3f}"
)
"""Best RF"""
best_rf = rf_search.best_estimator_
rf_test_pred = best_rf.predict(X_test)
rf_test_mae = mean_absolute_error(y_test, rf_test_pred)
rf_test_rmse = mean_squared_error(y_test, rf_test_pred)**0.5
rf_test_r2 = r2_score(y_test, rf_test_pred)
print("\nBest Random Forest Test:")
print(f"MAE: {rf_test_mae:.3f}")
print(f"R2: {rf_test_r2:.3f}")
print(f"RMSE: {rf_test_rmse:.3f}")
#slight win with 27.088

"""Gradient Boosted Trees"""
#try without One Hot Encoding

gb_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False
                          ),
            categorical_features
        )
    ]
)
gb = Pipeline(
    steps=[
        ("preprocessor", gb_preprocessor),
        (
            "model",
            HistGradientBoostingRegressor(
                learning_rate=0.5,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=42
            )
        )
    ]
)
gb.fit(X_train, y_train)
gb_train_pred = gb.predict(X_train)
print("\nGradient Boosting Training:")
print(
    f"MAE: "
    f"{mean_absolute_error(y_train, gb_train_pred):.3f}"
)
print(f"R2: {r2_score(y_train, gb_train_pred):.3f}")
print(f"RMSE: {mean_squared_error(y_train, gb_train_pred)**0.5:.3f}")
#promising results

gb_cv = cross_validate(
    gb,
    X_train,
    y_train,
    cv=5,
    scoring={
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2"
    },
    n_jobs=-1,
)

gb_cv_mae = -gb_cv["test_mae"].mean()
gb_cv_rmse = -gb_cv["test_rmse"].mean()
gb_cv_r2 = gb_cv["test_r2"].mean()

print("\nGradient Boosted Trees 5-Fold CV:")
print(f"MAE: {gb_cv_mae:.3f}")
print(f"R2: {gb_cv_r2:.3f}")
print(f"RMSE: {gb_cv_rmse:.3f}")

gb_param_grid = {
    "model__learning_rate": [
        0.03, 0.05, 0.1
    ],
    "model__max_leaf_nodes": [
        5, 10, 15
    ],
    "model__min_samples_leaf": [
        20, 40, 80
    ],
    "model__l2_regularization": [
        1.0, 5.0, 10.0
    ]
}
gb_search = GridSearchCV(
    estimator=gb,
    param_grid=gb_param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1,
)

gb_search.fit(X_train, y_train)
print("\nBest Gradient Boosted Trees 5-Fold CV:")
print(gb_search.best_params_)
print(f"Best GB CV MAE: {-gb_search.best_score_:.3f}")

best_gb = gb_search.best_estimator_
gb_test_ored = best_gb.predict(X_test)
gb_test_mae = mean_absolute_error(y_test, gb_test_ored)
gb_test_rmse = mean_squared_error(y_test, gb_test_ored)**0.5
gb_test_r2 = r2_score(y_test, gb_test_ored)
print("\nBest GBT:")
print(f"MAE: {gb_test_mae:.3f}")
print(f"R2: {gb_test_r2:.3f}")
print(f"RMSE: {gb_test_rmse:.3f}")