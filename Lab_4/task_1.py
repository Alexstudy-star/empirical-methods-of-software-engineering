import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main() -> None:
    # fetch data
    housing = fetch_california_housing()
    df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
    df['MedHouseVal'] = housing.target

    fig = plt.figure(figsize=(11, 6))

    # correlation heatmap
    corr = df.corr('pearson')

    axis = fig.add_subplot(2, 3, 1)
    sns.heatmap(data=corr, ax=axis, annot=True, fmt='.2f', annot_kws={'size': 7}, cmap='coolwarm')
    axis.tick_params(axis='x', labelrotation=45)

    # california housing map
    axis = fig.add_subplot(2, 3, 2)
    axis.scatter(df['Longitude'], df['Latitude'], c=df['MedHouseVal'], s=5, alpha=0.5)
    axis.grid()

    # top attributes
    top_attributes = corr['MedHouseVal'].abs().sort_values(ascending=False).iloc[1:4]
    print(top_attributes)

    # multiple linear regression
    x_attr, y_attr, z_attr = top_attributes.index

    X = df[[x_attr, y_attr, z_attr]]  # independent variable
    Y = df[['MedHouseVal']]  # dependent variable

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    model = LinearRegression()
    model.fit(x_train, y_train)

    axis = fig.add_subplot(2, 3, 3, projection='3d')
    axis.scatter(df[x_attr], df[y_attr], df[z_attr], c=df['MedHouseVal'], alpha=0.3)
    axis.set_xlabel(x_attr)
    axis.set_ylabel(y_attr)
    axis.set_zlabel(z_attr)

    # plotly
    scatter_plot = px.scatter_3d(data_frame=df, x=x_attr, y=y_attr, z=z_attr,
                                 color='MedHouseVal', opacity=0.4, color_continuous_scale=px.colors.sequential.Bluered)
    scatter_plot.update_traces(marker=dict(size=3))

    # linear regression
    Y = df[['MedHouseVal']]  # dependent variable

    for i, attribute in zip(range(4, 7), top_attributes.index):
        X = df[[attribute]]  # independent variable

        # model learning
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

        model = LinearRegression()
        model.fit(x_train, y_train)

        # scatter diagram
        axis = fig.add_subplot(2, 3, i)

        axis.scatter(X, Y, s=2, alpha=0.2)
        axis.set_xlabel(attribute)
        axis.set_ylabel('MedHouseVal')

        axis.plot(X, model.predict(X), color='red')

    # show diagrams
    scatter_plot.show()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
