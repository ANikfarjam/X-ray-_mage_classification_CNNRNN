import plotly.graph_objects as go
import pandas as pd

# Data
model_performance = {
    'Main Model': [88.5, 30.5],
    'CNN with Nas': [86.95, 49.74],
    'CNN': [51.70, 75.70],
    'RF': [69.26, None]
}

# Create DataFrame
metrics = ['Accuracy', 'Loss']
df = pd.DataFrame(model_performance, index=metrics)

# Calculate improvements of Main Model over others
improvements = {}
for model, values in model_performance.items():
    if model != 'Main Model':
        improvements[model] = [
            df['Main Model'][i] - value if value is not None else None
            for i, value in enumerate(values)
        ]

# Convert improvements into a DataFrame for table
improvements_df = pd.DataFrame(improvements, index=metrics)

# Create bar chart
fig = go.Figure()
for model in df.columns:
    fig.add_trace(go.Bar(name=model, x=metrics, y=df[model], text=df[model], textposition='auto'))

# Create table data for improvements
table_data = {
    "Metric": metrics,
}
for model in improvements.keys():
    table_data[model] = [f"{value:.2f}" if value is not None else "N/A" for value in improvements[model]]

# Create the table below the graph
fig.add_trace(
    go.Table(
        header=dict(values=["Metric"] + list(improvements.keys()), align="left"),
        cells=dict(values=[table_data[key] for key in table_data], align="left")
    )
)

# Update layout
fig.update_layout(
    title="Model Performance Comparison and Main Model Improvements",
    barmode='group',
    xaxis_title="Metrics",
    yaxis_title="Performance",
    legend_title="Models",
    height=700
)

fig.show()
