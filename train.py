from sklearn.tree import DecisionTreeRegressor
from misc import (
    load_data, get_features_target, split_data,
    build_pipeline, train_model, evaluate_model
)

def main():
    df = load_data()
    X, y = get_features_target(df, target_col="MEDV")
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(random_state=42)
    pipe = build_pipeline(model, feature_names)
    pipe = train_model(pipe, X_train, y_train)
    mse = evaluate_model(pipe, X_test, y_test)
    print(f"[DecisionTreeRegressor] Test MSE: {mse:.4f}")

if __name__ == "__main__":
    main()
