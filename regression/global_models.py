import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

def load_training_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RELATIVE_PATH = os.path.join('regression', 'training_data.txt')
    FULL_PATH = os.path.join(BASE_DIR, RELATIVE_PATH)
    return pd.read_csv(FULL_PATH, sep='\t')


def train_models(df):
    X = df[["Pe0", "A", "sigma", "N"]]
    y_ws = df["window_size"]
    y_ms = df["min_slope"]

    X_train, X_test, y_ws_train, y_ws_test = train_test_split(X, y_ws, test_size=0.2, random_state=42)
    _, _, y_ms_train, y_ms_test = train_test_split(X, y_ms, test_size=0.2, random_state=42)

    model_ws = GradientBoostingRegressor(random_state=42)
    model_ws.fit(X_train, y_ws_train)
    model_ms = GradientBoostingRegressor(random_state=42)
    model_ms.fit(X_train, y_ms_train)

    return model_ws, model_ms

df = load_training_data()
model_ws, model_ms = train_models(df)


def predict_params(Pe0, A, sigma, N, model_ws, model_ms):
    x_input = pd.DataFrame([[Pe0, A, sigma, N]], columns=["Pe0", "A", "sigma", "N"])
    ws = model_ws.predict(x_input)[0]
    ms = model_ms.predict(x_input)[0]
    return int(round(ws)), round(ms, 3)