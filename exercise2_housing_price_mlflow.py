import os
import tarfile

import numpy as np
import pandas as pd
import warnings
from six.moves import urllib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
import mlflow
import mlflow.sklearn

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = urllib.parse.urljoin("datasets/", "housing/")
HOUSING_URL = urllib.parse.urljoin(DOWNLOAD_ROOT, "datasets/housing/housing.tgz")

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = urllib.parse.urljoin(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    fetch_housing_data()
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def prepare_data(df_data):
        df_data_num = df_data.drop("ocean_proximity", axis=1)
        df_data_prepared = imputer.transform(df_data_num)
        df_data_prepared = pd.DataFrame( df_data_prepared, columns=df_data_num.columns, index=df_data.index)
        df_data_prepared["rooms_per_household"] = df_data_prepared["total_rooms"] / df_data_prepared["households"]
        df_data_prepared["bedrooms_per_room"] = df_data_prepared["total_bedrooms"] / df_data_prepared["total_rooms"]
        df_data_prepared["population_per_household"] = df_data_prepared["population"] / df_data_prepared["households"]
        df_data_cat = df_data[["ocean_proximity"]]
        dummy_var_temp = pd.get_dummies(df_data_cat, drop_first=True)
        df_data_prepared = df_data_prepared.join(dummy_var_temp)
        return df_data_prepared

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    housing = load_housing_data()
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    current_experiment=dict(mlflow.get_experiment_by_name("Housing_price_prediction"))
    experiment_id=current_experiment['experiment_id']
    with mlflow.start_run(
        run_name="PARENT_RUN",
        experiment_id=experiment_id,
        tags={"version": "v1", "priority": "P1"},
        description="parent",
    ) as parent_run:
        mlflow.log_param("parent", "yes")
        mlflow.log_artifact(os.path.join(HOUSING_PATH, "housing.csv"))

        with mlflow.start_run(
            run_name="CHILD_RUN_DATA_PREP",
            experiment_id=experiment_id,
            description="child_Data_Preparation",
            nested=True,
        ) as child_run_data_prep:
            housing = strat_train_set.drop( "median_house_value", axis=1) 
            housing_labels = strat_train_set["median_house_value"].copy()
            imputer = SimpleImputer(strategy="median")
            housing_num = housing.drop("ocean_proximity", axis=1)
            imputer.fit(housing_num)
            housing_prepared = prepare_data(housing)

        with mlflow.start_run(
            run_name="CHILD_RUN_Grid_RandomForest",
            experiment_id=experiment_id,
            description="child_GridSearchCV_Random_Forest_Regressor",
            nested=True,
        ) as child_run_grid_rf:
            param_grid = [
                {"n_estimators": [3, 10], "max_features": [2, 4, 6]},
                {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3]},
            ]
            forest_reg = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                forest_reg,
                param_grid,
                cv=5,
                scoring="neg_mean_squared_error",
                return_train_score=True,
            )
            grid_search.fit(housing_prepared, housing_labels)
            final_model = grid_search.best_estimator_
            grid_rf_rmse = np.sqrt(-(grid_search.best_score_))
            housing_predictions = final_model.predict(housing_prepared)
            grid_rf_mae = mean_absolute_error(housing_labels, housing_predictions)

            mlflow.log_param("child_Grid_random_forest", "yes")
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics({"mae": grid_rf_mae, "rmse": grid_rf_rmse})
            mlflow.sklearn.log_model(final_model, "model")


        with mlflow.start_run(
            run_name="CHILD_RUN_scoring",
            experiment_id=experiment_id,
            description="child_scoring",
            nested=True,
        ) as child_run_score:
            X_test = strat_test_set.drop("median_house_value", axis=1)
            X_test_prepared = prepare_data(X_test)
            y_test = strat_test_set["median_house_value"].copy()
            final_predictions = final_model.predict(X_test_prepared)
            final_mse = mean_squared_error(y_test, final_predictions)
            final_rmse = np.sqrt(final_mse)
            final_mae = mean_absolute_error(y_test, final_predictions)
            mlflow.log_param("child_scoring", "yes")
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics({"mae": final_mae, "rmse": final_rmse})
            mlflow.sklearn.log_model(final_model, "model")