from dash import Dash, dcc, html, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import time
import threading
import sqlite3
import plotly.express as px
from playsound import playsound

# Initialize the Dash app with a dark theme
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Initialize SQLite database connection
conn = sqlite3.connect('pipeline_data.db', check_same_thread=False)
cursor = conn.cursor()

# Create a table to store pipeline data if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS pipeline_log (
                    id INTEGER PRIMARY KEY,
                    sequence TEXT,
                    optimization_status TEXT,
                    score INTEGER,
                    pass_fail TEXT,
                    prediction_results TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()

# Load data from the database into a DataFrame
def load_data():
    df = pd.read_sql('SELECT * FROM pipeline_log', conn)
    return df

df = load_data()

# Function to simulate the asynchronous processing pipeline
def simulate_pipeline():
    while True:
        # Generate new data for simulation
        new_sequence = f'Seq-{len(df) + 1}'
        optimization_status = 'Pending'
        score = np.random.randint(0, 100)
        pass_fail = 'Fail' if score < 60 else 'Pass'
        prediction_results = 'N/A' if pass_fail == 'Fail' else 'Predicted'

        # Add new row to DataFrame and log to database
        new_row = {
            'sequence': new_sequence,
            'optimization_status': optimization_status,
            'score': score,
            'pass_fail': pass_fail,
            'prediction_results': prediction_results
        }
        df.loc[len(df)] = new_row

        cursor.execute('''INSERT INTO pipeline_log (sequence, optimization_status, score, pass_fail, prediction_results) 
                          VALUES (?, ?, ?, ?, ?)''', 
                       (new_sequence, optimization_status, score, pass_fail, prediction_results))
        conn.commit()

        # Simulate processing time
        time.sleep(3)

        # Update optimization status to 'Completed'
        df.at[len(df) - 1, 'optimization_status'] = 'Completed'
        cursor.execute('UPDATE pipeline_log SET optimization_status = "Completed" WHERE sequence = ?', (new_sequence,))
        conn.commit()

        # Trigger alert if sequence passes threshold
        if pass_fail == 'Pass':
            playsound('alert_sound.mp3')  # Play sound alert

# Run the pipeline simulation in a separate thread
threading.Thread(target=simulate_pipeline, daemon=True).start()

# App layout with dark theme and accent colors
app.layout = dbc.Container(
    [
        html.H1("Molecule Generation and Analysis Pipeline", style={'color': '#00C896'}),  # Green accent for title
        dcc.Interval(
            id='interval-component',
            interval=2*1000,  # Interval in milliseconds (2 seconds)
            n_intervals=0
        ),
        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id='pipeline-table',
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=df.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '10px',
                        'font_family': 'Arial',
                        'font_size': '16px',
                        'border': '1px solid #2D2D2D',
                        'backgroundColor': '#1F1F1F',  # Dark background
                        'color': '#E0E0E0'  # Light text
                    },
                    style_header={
                        'backgroundColor': '#2B2B2B',  # Dark header background
                        'fontWeight': 'bold',
                        'color': '#00C896'  # Green accent for header
                    },
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{pass_fail} = "Pass"'},
                            'backgroundColor': '#007BFF',  # Blue accent for Pass
                            'color': 'white'
                        },
                        {
                            'if': {'filter_query': '{pass_fail} = "Fail"'},
                            'backgroundColor': '#FF4B4B',  # Red accent for Fail
                            'color': 'white'
                        }
                    ],
                    row_selectable='single'
                ),
            ], width=8),
            dbc.Col([
                html.Div(id='details-sidebar', style={'display': 'none', 'backgroundColor': '#292929', 'padding': '15px'}),
            ], width=4)
        ]),
        dcc.Graph(id='score-trend-graph', style={'backgroundColor': '#1F1F1F', 'color': '#00C896'})  # Graph with dark background and green accent
    ],
    fluid=True,
    style={'backgroundColor': '#121212'}  # Dark theme for entire container
)

# Callback to update the table data in real-time
@app.callback(
    Output('pipeline-table', 'data'),
    Input('interval-component', 'n_intervals')
)
def update_table(n):
    global df
    df = load_data()  # Reload data from database
    return df.to_dict('records')

# Callback to show the pop-up and play sound when a sequence passes the threshold
@app.callback(
    Output("modal", "is_open"),
    [Input('interval-component', 'n_intervals')],
    [State("modal", "is_open")]
)
def display_alert(n, is_open):
    if any(df['pass_fail'] == 'Pass'):
        return not is_open
    return is_open

# Callback for interactive visualizations
@app.callback(
    Output('score-trend-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    fig = px.line(df, x='timestamp', y='score', title='Score Trends Over Time', markers=True)
    fig.update_layout(
        plot_bgcolor='#1F1F1F',  # Dark plot background
        paper_bgcolor='#121212',  # Dark paper background
        font_color='#00C896'  # Green accent for text
    )
    return fig

# Callback to update details sidebar based on selected row
@app.callback(
    [Output('details-sidebar', 'children'),
     Output('details-sidebar', 'style')],
    Input('pipeline-table', 'selected_rows')
)
def update_sidebar(selected_rows):
    if selected_rows:
        selected_index = selected_rows[0]
        selected_row = df.iloc[selected_index]
        details = [
            html.H4(f"Sequence: {selected_row['sequence']}", style={'color': '#00C896'}),
            html.P(f"Optimization Status: {selected_row['optimization_status']}", style={'color': '#FFFFFF'}),
            html.P(f"Score: {selected_row['score']}", style={'color': '#FFFFFF'}),
            html.P(f"Pass/Fail: {selected_row['pass_fail']}", style={'color': '#FFFFFF'}),
            html.P(f"Prediction Results: {selected_row['prediction_results']}", style={'color': '#FFFFFF'})
        ]
        style = {'display': 'block', 'padding': '10px', 'backgroundColor': '#292929', 'border': '1px solid #00C896'}
        return details, style
    
    return [], {'display': 'none'}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
