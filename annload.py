
import pandas as pd
import numpy as np

def load():
    dataset = pd.read_csv('03-File1.csv')
    dataset_power_zero = dataset[dataset['Date'] == '0']
    dataset = dataset.drop(dataset_power_zero.index, axis=0)
    values = dataset.iloc[:, 2:6].values
    values = values.astype('float32')
    X = values[:, 0:3]
    y = values[:, 3]

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    X_train = sc_x.fit_transform(X_train)
    X_test = sc_x.transform(X_test)


    return sc_x