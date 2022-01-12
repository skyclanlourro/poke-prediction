# -*- coding: utf-8 -*-
# librairy
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# df = pd.read_csv('./data/pokemon.csv')

#create function build_model => clf

def build_model(df, features):
    df = df.set_index("#")
    # cor = df.corr()
    # print(sns.heatmap(cor))

    X = df.filter(features)

    y = df.filter(["Type 1"])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0
    )

    categorical_cols = [
        cname for cname in X_train.columns if X_train[cname].dtype in ["object"]
    ]

    numerical_cols = [
        cname
        for cname in X_train.columns
        if X_train[cname].dtype in ["int64", "float64", "bool"]
    ]
    # Preprocess pour numerical data
    numerical_transformer = SimpleImputer(strategy="constant")

    # Preprocess pour categorical data
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = RandomForestClassifier(n_estimators=1000, random_state=0)


    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    clf.fit(X_train, y_train.values.ravel())

    acc = clf.score(X_valid, y_valid)

    return clf, acc
    
# --------------------------------------------------------------------------------
def type_prediction(pokemon_name, df, clf, features):
    print(df.columns)
    df = df.set_index("#")
    # cor = df.corr()
    # print(sns.heatmap(cor))
    # prédiction de type pokemon  depuis name
    X_to_predict = df.loc[df["Name"] == pokemon_name, features]
    print(X_to_predict.columns)
    y_value = df.loc[df["Name"] == pokemon_name, "Type 1"]
    predicted = clf.predict(X_to_predict)

    pokemon_type = predicted[0]

    fig = ""
    return pokemon_type, fig

# --------------------------------------------------------------------------------

    # save the model to disk
filename = 'finalized_model.pkl'
pickle.dump(build_model, open(filename, 'wb'))