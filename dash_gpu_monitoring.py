import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

# Initialize the Dash app
app = dash.Dash(__name__)


# Function to load and preprocess data
def load_data():
    df = pd.read_csv('data/results-file.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(r'\s*\[.*\]', '', regex=True)
    return df


# Define the layout
app.layout = html.Div([
    html.H1("GPU Monitoring Dashboard"),

    dcc.Dropdown(id='device-dropdown', multi=False),

    dcc.Dropdown(
        id='metric-dropdown',
        options=[
            {'label': 'Temperature', 'value': 'temperature.gpu'},
            {'label': 'GPU Utilization', 'value': 'utilization.gpu'},
            {'label': 'Memory Utilization', 'value': 'utilization.memory'}
        ],
        value='temperature.gpu',
        multi=False
    ),

    dcc.Graph(id='time-series-chart'),
    dcc.Graph(id='utilization-scatter'),
    dcc.Graph(id='memory-pie-chart'),

    dcc.Interval(
        id='interval-component',
        interval=30 * 1000,  # in milliseconds, update every 30 seconds
        n_intervals=0
    ),

    # Store the dataframe in a hidden div
    dcc.Store(id='stored-data')
])


# Callback to periodically load data
@app.callback(
    Output('stored-data', 'data'),
    Input('interval-component', 'n_intervals')
)
def update_data(n):
    df = load_data()
    return df.to_dict('records')


# Callback to update device dropdown options
@app.callback(
    Output('device-dropdown', 'options'),
    Output('device-dropdown', 'value'),
    Input('stored-data', 'data')
)
def update_device_dropdown(data):
    if data is None:
        raise PreventUpdate
    df = pd.DataFrame(data)
    options = [{'label': i, 'value': i} for i in df['name'].unique()]
    value = df['name'].unique()[0] if options else None
    return options, value


# Callback for time series chart
@app.callback(
    Output('time-series-chart', 'figure'),
    Input('device-dropdown', 'value'),
    Input('metric-dropdown', 'value'),
    Input('stored-data', 'data')
)
def update_time_series(selected_device, selected_metric, data):
    if data is None or selected_device is None:
        raise PreventUpdate
    df = pd.DataFrame(data)
    filtered_df = df[df['name'] == selected_device]
    fig = px.line(filtered_df, x='timestamp', y=selected_metric,
                  title=f'{selected_metric} Over Time for {selected_device}')
    return fig


# Callback for scatter plot
@app.callback(
    Output('utilization-scatter', 'figure'),
    Input('device-dropdown', 'value'),
    Input('stored-data', 'data')
)
def update_scatter(selected_device, data):
    if data is None or selected_device is None:
        raise PreventUpdate

    df = pd.DataFrame(data)
    filtered_df = df[df['name'] == selected_device]

    # Debug: Print column names and first few rows
    print("Columns:", filtered_df.columns)
    print("First few rows:")
    print(filtered_df.head())

    # Check if the required columns exist
    if 'utilization.gpu' not in filtered_df.columns or 'utilization.memory' not in filtered_df.columns:
        print("Required columns not found. Using placeholder data.")
        # Create a placeholder scatter plot
        fig = go.Figure(go.Scatter(x=[0], y=[0], mode='markers'))
        fig.update_layout(title=f"Data unavailable for {selected_device}")
        return fig

    # Check for non-numeric data
    filtered_df['utilization.gpu'] = pd.to_numeric(filtered_df['utilization.gpu'], errors='coerce')
    filtered_df['utilization.memory'] = pd.to_numeric(filtered_df['utilization.memory'], errors='coerce')

    # Remove rows with NaN values
    filtered_df = filtered_df.dropna(subset=['utilization.gpu', 'utilization.memory'])

    if filtered_df.empty:
        print("No valid data after cleaning.")
        fig = go.Figure(go.Scatter(x=[0], y=[0], mode='markers'))
        fig.update_layout(title=f"No valid data for {selected_device}")
        return fig

    try:
        fig = px.scatter(filtered_df, x='utilization.gpu', y='utilization.memory',
                         title=f'GPU vs Memory Utilization for {selected_device}')
        return fig
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
        # Create a simple scatter plot using graph_objects as a fallback
        fig = go.Figure(go.Scatter(x=filtered_df['utilization.gpu'],
                                   y=filtered_df['utilization.memory'],
                                   mode='markers'))
        fig.update_layout(title=f'GPU vs Memory Utilization for {selected_device}',
                          xaxis_title='GPU Utilization',
                          yaxis_title='Memory Utilization')
        return fig

# Callback for pie chart
@app.callback(
    Output('memory-pie-chart', 'figure'),
    Input('device-dropdown', 'value'),
    Input('stored-data', 'data')
)
def update_pie_chart(selected_device, data):
    if data is None or selected_device is None:
        raise PreventUpdate
    df = pd.DataFrame(data)
    filtered_df = df[df['name'] == selected_device].iloc[-1]
    values = [filtered_df['memory.used'], filtered_df['memory.free']]
    labels = ['Used', 'Free']
    fig = px.pie(values=values, names=labels,
                 title=f'Memory Usage for {selected_device}')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
