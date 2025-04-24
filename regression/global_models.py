from regression.optuna_search import train_models, load_training_data

df = load_training_data()
model_ws, model_ms = train_models(df)