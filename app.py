import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from dash import Dash, dcc, html, Input, Output

# Load and preprocess data
data = pd.read_csv("insurance.csv")
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['smoker'] = le.fit_transform(data['smoker'])
data['region'] = le.fit_transform(data['region'])

# Discretize 'expenses' into two classes
data['expenses'] = pd.cut(data['expenses'], bins=[0, 10000, float('inf')], labels=[0, 1])

# Split data
X = data.drop('expenses', axis=1)
y = data['expenses']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier()
}

# Define function to calculate metrics
def calculate_metrics(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }

# Create Dash app
app = Dash(__name__)
app.layout = html.Div([
    dcc.Dropdown(
        id='model-selector',
        options=[{'label': name, 'value': name} for name in models.keys()],
        value='Decision Tree'
    ),
    dcc.Graph(id='metrics-graph')
])

# Define callback to update graph
@app.callback(
    Output('metrics-graph', 'figure'),
    Input('model-selector', 'value')
)
def update_graph(selected_model):
    metrics = calculate_metrics(models[selected_model])
    return {
        'data': [{'x': list(metrics.keys()), 'y': list(metrics.values()), 'type': 'bar'}],
        'layout': {'title': 'Model Performance Metrics'}
    }

if __name__ == '__main__':
    app.run_server(debug=True)