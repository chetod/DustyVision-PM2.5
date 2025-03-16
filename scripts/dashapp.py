import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pycaret.regression import load_model, predict_model
from datetime import datetime, timedelta
import numpy as np

# Load data
file_path = r"C:\Users\ASUS\Desktop\projectforecastpm2_5\dataforecast\cleandata_vtest_hours.csv"
df = pd.read_csv(file_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
np.random.seed(42)
# Load the trained model
model = load_model(r'C:\Users\ASUS\Desktop\projectforecastpm2_5\models\best_model')

def forecast_next_7_days(model, last_data):
    last_date = last_data['timestamp'].max()
    future_hours = [last_date + timedelta(hours=i+1) for i in range(30 * 24)]  # 7 days * 24 hours
    
    future_data = []
    current_data = last_data.copy()
    
    for future_hour in future_hours:
        new_row = {'timestamp': future_hour}
        
        # Calculate mean temperature and humidity from recent data
        temp_mean = current_data['temperature'].tail(24).mean()
        humidity_mean = current_data['humidity'].tail(24).mean()
        
        # Generate temperature and humidity using normal distribution
        new_row['temperature'] = np.random.normal(temp_mean, 2)
        new_row['humidity'] = np.clip(np.random.normal(humidity_mean, 5), 0, 100)
        
        # Create Lag Features for PM2.5
        for lag in range(1, 8):
            if len(current_data) >= lag:
                new_row[f'pm_2_5_Lag{lag}'] = current_data['pm_2_5'].iloc[-lag]
            else:
                new_row[f'pm_2_5_Lag{lag}'] = current_data['pm_2_5'].mean()
        
        # Predict PM2.5 for this hour
        new_df = pd.DataFrame([new_row])
        prediction = predict_model(model, data=new_df)
        
        new_row['pm_2_5'] = prediction['prediction_label'].iloc[0]
        future_data.append(new_row)
        
        # Add predicted data to current_data for next hour prediction
        current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
    
    return pd.DataFrame(future_data)

# Generate forecast data
forecast_data = forecast_next_7_days(model, df)

# Add hour and date columns for easier filtering
forecast_data['date'] = forecast_data['timestamp'].dt.date
forecast_data['hour'] = forecast_data['timestamp'].dt.hour
forecast_data['day'] = forecast_data['timestamp'].dt.day_name()

# Define PM2.5 quality levels
def get_pm25_quality(value):
    if value <= 12.0:
        return "Good"
    elif value <= 35.4:
        return "Moderate"
    elif value <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif value <= 150.4:
        return "Unhealthy"
    elif value <= 250.4:
        return "Very Unhealthy"
    else:
        return "Hazardous"

forecast_data['quality'] = forecast_data['pm_2_5'].apply(get_pm25_quality)

# Define color mapping for PM2.5 quality
color_map = {
    "Good": "#00E400",
    "Moderate": "#FFFF00",
    "Unhealthy for Sensitive Groups": "#FF7E00",
    "Unhealthy": "#FF0000",
    "Very Unhealthy": "#8F3F97",
    "Hazardous": "#7E0023"
}

# Create Dash application with custom CSS
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ]
)

# Define custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>PM2.5 Forecast Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Roboto', sans-serif;
                margin: 0;
                background-color: #f5f5f5;
            }
            .main-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: white;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            .header {
                background: linear-gradient(135deg, #6c7ae0, #3e4491);
                color: white;
                padding: 20px;
                text-align: center;
                border-radius: 5px 5px 0 0;
                margin-bottom: 20px;
            }
            .card {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                padding: 15px;
                margin-bottom: 20px;
            }
            .chart-container {
                height: 400px;
            }
            .info-box {
                padding: 15px;
                background-color: #e3f2fd;
                border-left: 5px solid #2196f3;
                border-radius: 3px;
                margin-bottom: 20px;
            }
            .info-box h4 {
                margin-top: 0;
                color: #1565c0;
            }
            .footer {
                text-align: center;
                padding: 15px;
                color: #555;
                font-size: 0.9em;
                margin-top: 30px;
                border-top: 1px solid #ddd;
            }
            .quality-indicator {
                display: inline-block;
                width: 15px;
                height: 15px;
                border-radius: 50%;
                margin-right: 5px;
            }
            .quality-legend {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                margin: 10px 0;
            }
            .quality-item {
                display: flex;
                align-items: center;
                margin: 5px 10px;
            }
            .stat-card {
                background: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                text-align: center;
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }
            .stat-label {
                color: #666;
            }
            .stat-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .date-picker-container {
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Create the app layout
app.layout = html.Div(className='main-container', children=[
    html.Div(className='header', children=[
        html.H1("PM2.5 Forecast Dashboard", style={'margin-bottom': '5px'}),
        html.P("7-day prediction of PM2.5 levels with hourly breakdown")
    ]),
    
    html.Div(className='info-box', children=[
        html.H4("PM2.5 Air Quality Index"),
        html.Div(className='quality-legend', children=[
            html.Div(className='quality-item', children=[
                html.Div(className='quality-indicator', style={'background-color': color}) for color in color_map.values()
            ]),
            html.Div(className='quality-item', children=[
                html.Span(level, style={'margin-right': '15px'}) for level in color_map.keys()
            ])
        ])
    ]),
    
    html.Div(className='stat-grid', children=[
        html.Div(className='stat-card', children=[
            html.I(className="fas fa-calendar-alt", style={'color': '#3e4491', 'font-size': '24px'}),
            html.Div(className='stat-value', id='forecast-days'),
            html.Div(className='stat-label', children="Days Forecasted")
        ]),
        html.Div(className='stat-card', children=[
            html.I(className="fas fa-wind", style={'color': '#3e4491', 'font-size': '24px'}),
            html.Div(className='stat-value', id='avg-pm25'),
            html.Div(className='stat-label', children="Average PM2.5")
        ]),
        html.Div(className='stat-card', children=[
            html.I(className="fas fa-exclamation-triangle", style={'color': '#3e4491', 'font-size': '24px'}),
            html.Div(className='stat-value', id='max-pm25'),
            html.Div(className='stat-label', children="Max PM2.5")
        ]),
        html.Div(className='stat-card', children=[
            html.I(className="fas fa-check-circle", style={'color': '#3e4491', 'font-size': '24px'}),
            html.Div(className='stat-value', id='min-pm25'),
            html.Div(className='stat-label', children="Min PM2.5")
        ])
    ]),
    
    html.Div(className='card', children=[
        html.H3("Daily PM2.5 Forecast", style={'margin-top': '0'}),
        html.P("Click on a day to see hourly breakdown", style={'color': '#666'}),
        dcc.Graph(id='7-day-forecast', className='chart-container')
    ]),
    
    html.Div(className='card', children=[
        html.H3("Hourly PM2.5 Breakdown", style={'margin-top': '0'}),
        html.P("Select a date to view hourly forecast", style={'color': '#666'}),
        html.Div(className='date-picker-container', children=[
            dcc.DatePickerSingle(
                id='date-picker',
                min_date_allowed=forecast_data['date'].min(),
                max_date_allowed=forecast_data['date'].max(),
                initial_visible_month=forecast_data['date'].min(),
                date=forecast_data['date'].min()
            )
        ]),
        dcc.Graph(id='hourly-forecast', className='chart-container')
    ]),
    
    html.Div(className='footer', children=[
        html.P("PM2.5 Forecast Dashboard © 2025"),
        html.P("Powered by PyCaret and Dash")
    ])
])

# Callback for statistics
@app.callback(
    [Output('forecast-days', 'children'),
     Output('avg-pm25', 'children'),
     Output('max-pm25', 'children'),
     Output('min-pm25', 'children')],
    [Input('date-picker', 'date')]
)
def update_stats(date):
    days = len(forecast_data['date'].unique())
    avg_pm25 = f"{forecast_data['pm_2_5'].mean():.1f}"
    max_pm25 = f"{forecast_data['pm_2_5'].max():.1f}"
    min_pm25 = f"{forecast_data['pm_2_5'].min():.1f}"
    return days, avg_pm25, max_pm25, min_pm25

# Callback for 7-day forecast chart
@app.callback(
    Output('7-day-forecast', 'figure'),
    [Input('date-picker', 'date')]
)
def update_7_day_forecast(date):
    # Group by date and calculate daily averages
    daily_forecast = forecast_data.groupby('date').agg({
        'pm_2_5': 'mean',
        'quality': lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
    }).reset_index()
    
    # Create a color list based on the quality values
    colors = [color_map[quality] for quality in daily_forecast['quality']]
    
    # Create figure with custom marker colors
    fig = go.Figure()
    
    # Add line and markers
    fig.add_trace(go.Scatter(
        x=daily_forecast['date'],
        y=daily_forecast['pm_2_5'],
        mode='lines+markers',
        name='PM2.5',
        line=dict(width=3, color='#3e4491'),
        marker=dict(size=12, color=colors, line=dict(width=2, color='#3e4491'))
    ))
    
    # Add quality annotations
    for i, row in daily_forecast.iterrows():
        fig.add_annotation(
            x=row['date'],
            y=row['pm_2_5'],
            text=f"{row['pm_2_5']:.1f}",
            showarrow=False,
            yshift=15,
            font=dict(size=12, color="#3e4491")
        )
    
    # Customize layout
    fig.update_layout(
        title=None,
        xaxis_title="Date",
        yaxis_title="PM2.5 (μg/m³)",
        template="plotly_white",
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=20, b=40),
        height=400
    )
    
    return fig

# Callback for hourly forecast chart
@app.callback(
    Output('hourly-forecast', 'figure'),
    [Input('date-picker', 'date')]
)
def update_hourly_forecast(date):
    if date is None:
        date = forecast_data['date'].min()
    
    # Convert string date to datetime.date
    if isinstance(date, str):
        selected_date = datetime.strptime(date, '%Y-%m-%d').date()
    else:
        selected_date = date
    
    # Filter data for selected date
    hourly_data = forecast_data[forecast_data['date'] == selected_date].copy()
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart for PM2.5 levels
    fig.add_trace(go.Bar(
        x=hourly_data['hour'],
        y=hourly_data['pm_2_5'],
        name='PM2.5',
        marker_color=[color_map[q] for q in hourly_data['quality']],
        hovertemplate='Hour: %{x}<br>PM2.5: %{y:.1f} μg/m³<br>Quality: %{text}<extra></extra>',
        text=hourly_data['quality']
    ))
    
    # Add line for better visualization
    fig.add_trace(go.Scatter(
        x=hourly_data['hour'],
        y=hourly_data['pm_2_5'],
        mode='lines',
        line=dict(color='#3e4491', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Customize layout
    day_name = hourly_data['day'].iloc[0] if not hourly_data.empty else ""
    fig.update_layout(
        title=f"Hourly PM2.5 Forecast for {day_name}, {selected_date}",
        xaxis=dict(
            title="Hour of Day",
            tickmode='array',
            tickvals=list(range(0, 24, 2)),  # Show every 2 hours
            ticktext=[f"{h}:00" for h in range(0, 24, 2)]  # Format as HH:00
        ),
        yaxis_title="PM2.5 (μg/m³)",
        template="plotly_white",
        hoverlabel=dict(bgcolor="white", font_size=12),
        barmode='relative',
        margin=dict(l=40, r=40, t=80, b=40),
        height=400
    )
    
    # Add horizontal reference lines for air quality levels
    reference_levels = [
        (12.0, "Good", "#00E400"),
        (35.4, "Moderate", "#FFFF00"),
        (55.4, "Unhealthy for Sensitive Groups", "#FF7E00"),
        (150.4, "Unhealthy", "#FF0000"),
        (250.4, "Very Unhealthy", "#8F3F97")
    ]
    
    for level, label, color in reference_levels:
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=level,
            x1=23.5,
            y1=level,
            line=dict(color=color, width=1, dash="dash"),
        )
        fig.add_annotation(
            x=23.5,
            y=level,
            xref="x",
            yref="y",
            text=label,
            showarrow=False,
            xanchor="right",
            font=dict(size=10, color=color),
            bgcolor="rgba(255, 255, 255, 0.7)",
            borderpad=2
        )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)