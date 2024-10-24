import pandas as pd
import numpy as np
from future_portfolio import (
    CustomDataLoader, 
    FeatureEngineer, 
    ModelTrainer, 
    PortfolioManager, 
    BacktestEngine, 
    rolling_window_backtest
)

def main(csv_file_path):
    data = CustomDataLoader.load_csv_data(csv_file_path)
    data = FeatureEngineer.create_features(data)

    features = ['SPX_Returns', 'SPX_MA50', 'SPX_MA200', 'SPX_Volatility', 'SPX_Returns_Lag1', 'SPX_Returns_Lag2', 'SPX_RSI', 'SPX_MACD']
    target = 'SPX_Returns'

    train_data = data[data.index < '2019-01-01']
    test_data = data[data.index >= '2019-01-01']

    # Run rolling window backtest
    train_results = rolling_window_backtest(train_data, test_data, features, target)

    # Print rolling window results
    for model, model_results in train_results.items():
        avg_return = np.mean([r[0] for r in model_results])
        avg_transactions = np.mean([r[1] for r in model_results])
        print(f"Model: {model}")
        print(f"Average Rolling Window Portfolio Return: {avg_return:.2%}")
        print(f"Average Rolling Window Number of Transactions: {avg_transactions:.2f}")
        print("--------------------")

    # Final training and testing
    model_trainer = ModelTrainer()
    model_trainer.train_models(train_data[features], train_data[target])
    test_predictions = model_trainer.predict(test_data[features])

    # Evaluate final performance on 2019-2023 test data
    for name, preds in test_predictions.items():
        pm = PortfolioManager()
        for pred, actual in zip(preds, test_data[target]):
            pm.make_decision(pred, actual)
        print(f"Model {name} final performance on 2019-2023 test data:")
        print(f"Portfolio Return: {pm.get_portfolio_return():.2%}")
        print(f"Number of Transactions: {pm.transactions}")
        print("--------------------")

    # Calculate and print benchmark return for 2019-2023
    spx_return = (test_data['SPX'].iloc[-1] / test_data['SPX'].iloc[0]) - 1
    print(f"Benchmark (Buy and Hold SPX) 2019-2023: {spx_return:.2%}")

if __name__ == "__main__":
    csv_file_path = "SPX_index.csv"  # Replace with the actual path to your CSV file
    main(csv_file_path)