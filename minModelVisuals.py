import pandas as pd
# import matplotlib.pyplot as pl
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix
#from crossForest import best_model
import tensorflow as tf
#from traditionalCNN import model
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp


main_model = tf.keras.models.load_model('Main_model_saved.h5')
#rf = best_model
#cnn = model


# Load data from CSV files
confusion_matrix = pd.read_csv("confusion_matrix.csv", header=None).values
classification_report = pd.read_csv("classification_report.csv")

# Extract metrics from the classification report
precision = classification_report['precision'].values[:-1]  # Exclude the last row (average)
recall = classification_report['recall'].values[:-1]
f1_score = classification_report['f1-score'].values[:-1]
support = classification_report['support'].values[:-1]

# Create subplots for visualization
fig = sp.make_subplots(
    rows=2, cols=2,
    subplot_titles=("Confusion Matrix", "Precision by Class", "Recall by Class", "F1-Score by Class"),
    specs=[[{"type": "heatmap"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]]
)

# Add confusion matrix as a heatmap
fig.add_trace(
    go.Heatmap(
        z=confusion_matrix,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="Count"),
    ),
    row=1, col=1
)

# Add precision bar chart
fig.add_trace(
    go.Bar(
        x=[f"Class {i}" for i in range(len(precision))],
        y=precision,
        name="Precision",
        marker_color="blue"
    ),
    row=1, col=2
)

# Add recall bar chart
fig.add_trace(
    go.Bar(
        x=[f"Class {i}" for i in range(len(recall))],
        y=recall,
        name="Recall",
        marker_color="green"
    ),
    row=2, col=1
)

# Add F1-score bar chart
fig.add_trace(
    go.Bar(
        x=[f"Class {i}" for i in range(len(f1_score))],
        y=f1_score,
        name="F1-Score",
        marker_color="orange"
    ),
    row=2, col=2
)

# Update layout for better visualization
fig.update_layout(
    title_text="Main Model Performance Visualization",
    height=800, width=1000,
    showlegend=False
)

# Show the plot
fig.show()

# Save the plot as an HTML file (optional)
fig.write_html("main_model_performance.html")