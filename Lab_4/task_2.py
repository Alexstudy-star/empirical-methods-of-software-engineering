import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error


def main() -> None:
    # fetch data
    housing = fetch_california_housing()
    df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
    df['MedHouseVal'] = housing.target

    fig = plt.figure(figsize=(10, 7))

    # top attributes
    corr = df.corr('pearson')
    top = 3
    top_attributes = corr['MedHouseVal'].abs().sort_values(ascending=False).iloc[1: top + 1]

    # data normalization
    scaler1 = MinMaxScaler()
    scaler1.fit(df)
    normalized_ndarray = scaler1.transform(df)
    df = pd.DataFrame(normalized_ndarray, columns=df.columns)

    # linear regression
    regression_types = [LinearRegression, Lasso, Ridge, ElasticNet]
    rows = len(top_attributes) + 1
    cols = len(regression_types)

    X = df[top_attributes.index]  # independent variable
    Y = df[['MedHouseVal']]  # dependent variable

    for i, (col_name, series) in enumerate(X.items()):
        col_data = pd.DataFrame(series)

        # split data
        x_train, x_test, y_train, y_test = train_test_split(col_data, Y, test_size=0.3)

        for j, regression in enumerate(regression_types):
            # model learning
            model = regression()
            model.fit(x_train, y_train)

            # scatter diagram
            axis = fig.add_subplot(rows, cols, (i * cols) + j + 1)

            axis.scatter(col_data, Y, s=2, alpha=0.2)
            axis.set_xlabel(col_name)

            if j == 0:
                axis.set_ylabel('MedHouseVal')
            if i == 0:
                axis.set_title(regression.__name__)

            axis.plot(col_data, model.predict(col_data), color='red')

    # split data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    # prediction
    results = list()

    for j, regression in enumerate(regression_types):
        # model learning
        model = regression() if regression == LinearRegression else regression(alpha=0.001)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # metrics
        R2 = r2_score(y_test, y_pred)
        MSE = mean_squared_error(y_test, y_pred)
        results.append({'Regression': regression.__name__, 'R2': R2, 'MSE': MSE})

        # scatter diagram
        axis = fig.add_subplot(rows, cols, ((rows-1) * cols) + j + 1)

        axis.scatter(y_test, y_pred, s=2, alpha=0.2)
        axis.set_xlabel('real data')

        if j == 0:
            axis.set_ylabel('prediction')

        axis.plot(y_test, y_test, color='red')

    results_df = pd.DataFrame(results)
    print(results_df)

    # show diagrams
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
