import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def preprocess_data(data):
    data.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    return data

def preprocess_time_series(data):
    data = data[["Close"]]
    data.reset_index(inplace=True)
    data.columns = ['ds', 'y']
    return data

def train_random_forest(data, predictors, target):
    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    model.fit(data[predictors], data[target])
    return model

def predict(model, data):
    predictions = model.predict(data)
    return predictions

def visualize_predictions(actual, predicted):
    combined = pd.concat([actual, predicted], axis=1)
    combined.plot()
    plt.savefig('plot.png')
    plt.close()

def main():
    tata_steel = yf.Ticker("TATASTEEL.NS").history(period="Max")
    tata_steel = preprocess_data(tata_steel)
    ts_data = preprocess_time_series(tata_steel)
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    target = "Target"
    train_data = tata_steel.iloc[:-100]
    test_data = tata_steel.iloc[-100:]
    model_rf = train_random_forest(train_data, predictors, target)
    next_data = {
        "Close": 131.85, "Volume": 34669506, "Open": 131.7, "High": 140, "Low": 100
    }
    next_data_point = pd.DataFrame([next_data])
    test_predictions = model_rf.predict(test_data[predictors])
    time_series_prediction = model_rf.predict(next_data_point[predictors])
    classification_prediction = model_rf.predict(next_data_point[predictors])
    visualize_predictions(test_data["Target"], pd.Series(test_predictions, index=test_data.index))
    accuracy = accuracy_score(test_data["Target"], test_predictions)
    print("Accuracy:", accuracy)

    print("Time Series Prediction:", time_series_prediction)
    print("Classification Prediction:", classification_prediction)

if __name__ == "__main__":
    main()
