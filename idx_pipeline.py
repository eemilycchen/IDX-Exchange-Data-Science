"""Shared pieces for the modeling notebooks (03, 04, 05).

02_preprocessing writes Dataset A: fixed cleaning rules, NaN where a value is missing or
invalid, and a <column>_missing flag for each. Everything that is LEARNED from data
(medians, modes, coordinate lookups, encoders, scalers) lives here, inside a sklearn
Pipeline, so it is fit on train only, or on the train part of each CV fold.
"""
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler, TargetEncoder)

SEED = 42
DATA_DIR = os.path.join("dataset", "processed")
TARGET = "ClosePrice"

# Must match 02_preprocessing
NUM_IMPUTE = ["LivingArea", "LotSizeSquareFeet", "GarageSpaces", "ParkingTotal", "YearBuilt",
              "BedroomsTotal", "BathroomsTotalInteger", "Stories", "AgeAtSale"]
YN_COLS = ["ViewYN", "PoolPrivateYN", "FireplaceYN", "NewConstructionYN", "AttachedGarageYN"]
CAT_COLS = ["City", "CountyOrParish", "PostalCode"]
# right-skewed sizes: log1p before scaling for linear models
LOG_COLS = ["LivingArea", "LotSizeSquareFeet", "GarageSpaces"]  # AssociationFee on hold until 05


def load_split(split, data_dir=DATA_DIR):
    # PostalCode is already a 5-digit string from 02; read it as text so leading zeros stay
    df = pd.read_csv(os.path.join(data_dir, f"{split}_clean.csv"), dtype={"PostalCode": str}, low_memory=False)
    df["CloseDate"] = pd.to_datetime(df["CloseDate"])
    return df


def metrics(y_true, y_pred):
    ape = np.abs(y_pred - y_true) / y_true  # absolute percentage error per home
    return {
        "MdAPE_%": round(100 * np.median(ape), 2),  # headline: error on the typical home
        "MAPE_%": round(100 * np.mean(ape), 2),
        "R2": round(r2_score(y_true, y_pred), 4),
        "MAE_$": round(mean_absolute_error(y_true, y_pred)),
        "within_10%": round(100 * np.mean(ape <= 0.10), 1),  # share of homes predicted within 10%
    }


def error_by(df, y_pred, groups):
    # metrics per group (e.g. price band or county)
    out = df.assign(pred=y_pred).groupby(groups, observed=True)
    rows = {g: {"n": len(d), **metrics(d[TARGET].to_numpy(), d["pred"].to_numpy())} for g, d in out}
    return pd.DataFrame(rows).T


class DatasetAImputer(BaseEstimator, TransformerMixin):
    """Dataset A fill rules (data-preprocessing.md), learned in fit:
    - numeric columns: train median (section 2, 3)
    - Y/N columns: train mode (section 5)
    - City / CountyOrParish / PostalCode: 'Unknown' (section 6)
    - Latitude / Longitude: median by PostalCode, then by CountyOrParish, then overall (section 4)
    The <column>_missing flags already come from 02, so this only fills values.
    Takes and returns a DataFrame; columns it doesn't know pass through unchanged.
    """

    def fit(self, X, y=None):
        self.medians_ = {c: float(X[c].median()) for c in NUM_IMPUTE if c in X}
        self.modes_ = {c: float(X[c].mode().iloc[0]) for c in YN_COLS if c in X}
        if {"Latitude", "Longitude"} <= set(X.columns):
            has = X["Latitude"].notna() & X["Longitude"].notna()
            coords = X.loc[has]
            self.zip_coords_ = coords.groupby("PostalCode")[["Latitude", "Longitude"]].median()
            self.county_coords_ = coords.groupby("CountyOrParish")[["Latitude", "Longitude"]].median()
            self.global_coords_ = coords[["Latitude", "Longitude"]].median()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        # coordinates first, while PostalCode / County still have NaN (not 'Unknown') where missing
        if hasattr(self, "zip_coords_"):
            for c in ["Latitude", "Longitude"]:
                fill = X["PostalCode"].map(self.zip_coords_[c])
                fill = fill.fillna(X["CountyOrParish"].map(self.county_coords_[c]))
                X[c] = X[c].fillna(fill).fillna(self.global_coords_[c])
        for c, v in self.medians_.items():
            X[c] = X[c].fillna(v)
        for c, v in self.modes_.items():
            X[c] = X[c].fillna(v)
        for c in CAT_COLS:
            if c in X:
                X[c] = X[c].fillna("Unknown").astype(str)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def build_preprocessor(kind, num_cols, cat_cols, log_cols=()):
    """Imputer + encoding, fit on train only.
    kind:
      "linear"  - log1p + scale log_cols, scale num_cols, one-hot cat_cols (rare -> 'infrequent')
      "tree"    - no scaling; cat_cols as integer codes, unseen categories -> -1
      "tree_te" - no scaling; cat_cols target-encoded. TargetEncoder cross-fits on train
                  (5 folds), so a row's encoding never sees its own price.
    """
    num_cols, cat_cols, log_cols = list(num_cols), list(cat_cols), list(log_cols)
    if kind == "linear":
        encode = ColumnTransformer([
            ("log", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                              ("scale", StandardScaler())]), log_cols),
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30), cat_cols),
        ])
    elif kind == "tree":
        encode = ColumnTransformer([
            ("num", "passthrough", log_cols + num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ])
    elif kind == "tree_te":
        encode = ColumnTransformer([
            ("num", "passthrough", log_cols + num_cols),
            ("cat", TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=SEED)),
             cat_cols),
        ])
    else:
        raise ValueError(f"unknown kind: {kind}")
    return Pipeline([("impute", DatasetAImputer()), ("encode", encode)])
