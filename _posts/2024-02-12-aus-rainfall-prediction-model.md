```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
```

# 1. Import the Dataset


```python
df = pd.read_csv("weatherAUS.csv")
df
```

# 2. Analyse the Dataset


```python
df.info()
```


```python
import numpy as np

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes('object').columns.tolist()
```


```python
numeric_cols
```


```python
categorical_cols
```


```python
# Checking how many values each column have.

df.count().sort_values()
```


```python
missing_values = df.isna().sum().sort_values()
num_missing_values = pd.DataFrame({'num_missing_values': missing_values})
print(num_missing_values)
```


```python
missing_percent = df.isnull().sum() * 100 / len(df)
missing_values = pd.DataFrame({'missing_values(%)': missing_percent})
missing_values.sort_values(by ='missing_values(%)' , ascending=False)
```

# 3. Preprocessing the Data

### 3.1 Discard unnecessary data

Removing rows with missing "RainToday" or "RainTomorrow" values before preprocessing can be a good idea to make analysis and modeling simpler and faster. Because I set "RainTomorrow" as the target and the other columns should be related to precipitation records.


```python
df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)
```

There are two ways to deal with missing values, either by deleting incomplete variables if there are too many data missing or by replacing these missing values with estimated value based on the other information available. So, any columns with more than 30% of missing values will be discarded and rest of the missing values will be replaced. Then before replaceing missing values of other columns, it's wise to first check for outliers to prevent causing errors while experimenting and entering data. In addition, it works better if the data is normally-distributed, while median imputation is preferable for skewed distribution. So, let's look through the result of data analysis with these considerations.


```python
numerical_A = ['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am']
df[numerical_A].hist(numerical_A, rwidth=0.7, figsize=(15, 10))
```


```python
df = df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am'])
```

### 3.2 Replace missing values


```python
numerical_B = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
               'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']
df[numerical_B].hist(numerical_B, rwidth=0.7, figsize=(20, 15))
```


```python
df[numerical_B].describe()
```


```python
# fill missing values of normally-distributed columns with mean and skewed distribution with median

df['MinTemp'] = df['MinTemp'].fillna(value = df['MinTemp'].mean())
df['MaxTemp'] = df['MaxTemp'].fillna(value = df['MaxTemp'].median())
df['Rainfall'] = df['Rainfall'].fillna(value = df['Rainfall'].median())
df['WindGustSpeed'] = df['WindGustSpeed'].fillna(value = df['WindGustSpeed'].median())
df['WindSpeed9am'] = df['WindSpeed9am'].fillna(value = df['WindSpeed9am'].median())
df['WindSpeed3pm'] = df['WindSpeed3pm'].fillna(value = df['WindSpeed3pm'].median())
df['Humidity9am'] = df['Humidity9am'].fillna(value = df['Humidity9am'].median())
df['Humidity3pm'] = df['Humidity3pm'].fillna(value = df['Humidity3pm'].median())
df['Pressure9am'] = df['Pressure9am'].fillna(value = df['Pressure9am'].median())
df['Pressure3pm'] = df['Pressure3pm'].fillna(value = df['Pressure3pm'].median())
df['Temp9am'] = df['Temp9am'].fillna(value = df['Temp9am'].median())
df['Temp3pm'] = df['Temp3pm'].fillna(value = df['Temp3pm'].median())
```

### 3.3 Handling categorical variables

It is well known that categorical data doesn't work with machine learning and deep learning algorithms, so i encoded 'Date', 'Location', 'RainToday' and 'RainTomorrow' columns so we can predict whether or not is going to rain tomorrow.


```python
s = (df.dtypes == "object")
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
```


```python
from datetime import datetime

# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Encode categorical variables
categorical_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']

for column in categorical_columns:
    df[column] = df[column].astype('category').cat.codes

# Encode 'RainToday' & 'RainTomorrow'
df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)
```


```python
# Check for confirming there no missing values

missing_percent = df.isnull().sum() * 100 / len(df)
missing_values = pd.DataFrame({'missing_values(%)': missing_percent})
missing_values.sort_values(by ='missing_values(%)' , ascending=False)
```


```python
df.info()
```

# 4. Data Preparation


```python
import plotly.express as px

fig = px.histogram(df, 
             x='RainToday', 
             color='RainTomorrow', 
             title='Rain Tomorrow vs. Rain Today')
fig.update_layout(width=500, height=400, bargap=0.2)
fig.show()
```


```python
px.scatter(df.sample(2000),
           x='Temp9am', 
           y='Temp3pm', 
           color='RainTomorrow')
```


```python
px.scatter(df.sample(2000),
           x='Humidity9am', 
           y='Humidity3pm', 
           color='RainTomorrow')
```


```python
px.scatter(df.sample(3000),
           x='WindGustSpeed', 
           y='Pressure3pm', 
           color='RainTomorrow')
```


```python
px.scatter(df.sample(100000),
           x='WindGustDir', 
           y='Rainfall', 
           color='RainTomorrow')
```


```python
px.scatter(df.sample(10000),
           x='WindSpeed3pm', 
           y='Pressure3pm', 
           color='RainTomorrow')
```


```python
px.scatter(df.sample(2000),
           x='MaxTemp', 
           y='MinTemp', 
           color='RainTomorrow')
```


```python
px.scatter(df.sample(2000),
           x='Pressure9am', 
           y='Pressure3pm', 
           color='RainTomorrow')
```


```python
px.scatter(df.sample(2000),
           x='Pressure9am',
           y='Rainfall', 
           color='RainTomorrow')
```


```python
features = df[['Temp9am', 'Temp3pm', 'Humidity9am', 'Humidity3pm', 'WindGustSpeed', 'MaxTemp', 'MinTemp', 
               'Rainfall', 'RainToday', 'Pressure9am', 'Pressure3pm', 'Location', 'Year', 'Month', 'Day']]

target = df['RainTomorrow']

features = (features - features.mean()) / features.std()
```

### 4.1 Training, Validation, and Test sets


```python
# Split ratios
split_ratio = 0.8

# Split index for training
split_index_train = int(len(features) * split_ratio)

X = features.values
y = target.values

# Training and Test sets
X_train = (X[:split_index_train] - X[:split_index_train].mean()) / X[:split_index_train].std()
X_test = (X[split_index_train:] - X[:split_index_train].mean()) / X[:split_index_train].std()

# Split index for validation (10%)
split_index_val = int(split_index_train * 0.1)

# Further split the training set into training and validation sets
X_train, X_val = X_train[:-split_index_val], X_train[-split_index_val:]
y_train, y_val = y[:split_index_train - split_index_val], y[split_index_train - split_index_val:split_index_train]
```

### 4.2 Convert data to PyTorch tensors


```python
# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y[split_index_train:], dtype=torch.float32)

# Create DataLoader for training set
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create DataLoader for validation set
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Create DataLoader for test set
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

### 4.3 Modeling


```python
import torch.nn.functional as F

class LogisticNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(LogisticNeuralNetwork, self).__init__()

        self.hidden_layers = nn.ModuleList([
            nn.Linear(input_size, hidden_sizes[0]),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Linear(hidden_sizes[1], hidden_sizes[2])
        ])

        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_sizes[2], output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Forward pass through hidden layers with ReLU activation
        for hidden_layer in self.hidden_layers:
            x = self.relu(hidden_layer(x))

        # Output layer with sigmoid activation
        out = self.output_layer(x)
        out = self.sigmoid(out)

        return out

input_size = X_train.shape[1]
hidden_sizes = [32, 16, 8]
output_size = 1
model = LogisticNeuralNetwork(input_size, hidden_sizes, output_size)
```

### 4.4 Loss Function and Optimizer


```python
# Loss Function and Optimizer
loss_fn = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.001)
```

### 4.5 Training & Validating


```python
# Train the Model
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for inputs, target in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, target.view(-1, 1))
        loss.backward()
        optimizer.step()

    # Validation during training
    model.eval()
    with torch.no_grad():
        val_loss = 0
        correct_val = 0
        total_val = 0

        for inputs, target in val_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, target.view(-1, 1))
            val_loss += loss.item()

            predicted = (outputs >= 0.5).float()
            total_val += target.size(0)
            correct_val += (predicted == target.view(-1, 1)).sum().item()

        # Test after each epoch
        test_loss = 0
        correct_test = 0
        total_test = 0

        model.eval()
        with torch.no_grad():
            for inputs, target in test_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, target.view(-1, 1))
                test_loss += loss.item()

                predicted = (outputs >= 0.5).float()
                total_test += target.size(0)
                correct_test += (predicted == target.view(-1, 1)).sum().item()

        if (epoch + 1) % 10 == 0:
            val_accuracy = correct_val / total_val
            test_accuracy = correct_test / total_test
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2%}, Test Accuracy: {test_accuracy:.2%}')
```

# 5. Evaluation


```python
from sklearn.metrics import confusion_matrix, classification_report

classes = ['No Rain', 'Raining']

y_pred = model(X_test)
y_pred = y_pred.ge(0.5).view(-1).cpu()
y_test = y_test.cpu()

# Create a classification report
report = classification_report(y_test, y_pred, target_names=classes)

# Print the entire classification report
print(report)

# Extract and print the test accuracy
test_accuracy = float(report.split()[-2])
print(f'Test Accuracy: {test_accuracy:.2%}')
```


```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

conf_mat = confusion_matrix(y_test, y_pred, normalize='pred')
classes = ['No Rain', 'Raining']
df_conf_mat = pd.DataFrame(conf_mat, index=classes, columns=classes)

plt.figure(figsize=(8, 6))
heat_map = sns.heatmap(df_conf_mat, annot=True, fmt='.2%', cmap='Blues')
heat_map.yaxis.set_ticklabels(heat_map.yaxis.get_ticklabels(), ha='right')
heat_map.xaxis.set_ticklabels(heat_map.xaxis.get_ticklabels(), ha='right')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix (% of Predictions)')
plt.show()
```
