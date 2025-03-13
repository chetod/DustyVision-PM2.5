import dash
from dash import dcc, html, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np


# อ่านข้อมูลจากไฟล์ CSV
df = pd.read_csv('pm25_data.csv')

# จากนั้นคุณสามารถใช้ df ใน Dash app ได้ตามปกติ
# Create sample data for PM2.5 forecasts
# From the image data, I'll create a structured dataframe
stations = [
    "บ้านหัวฝาย", "บ้านทุ่ง", "พุทธชาด", "วังเจ้า", "สก.วะกรีย์", 
    "สมเด็จพระเจ้าสุราษธานีสินมหาราช", "สามเงา", "อุ้มผาง", "แม่ระมาด", "แม่สอด"
]

# Generate dates for the next week
base_date = pd.Timestamp('2025-03-12')
dates = [(base_date + pd.Timedelta(days=i)).strftime('%d-%m-%Y') for i in range(7)]

# Extract data from the image
data = {
    'station': stations,
    'today': [37, 25, 17, 40, 43, 28, 49, 41, 129, 81],
    'day1': [20, 25, None, 19, None, 22, 22, 11, 42, 20],
    'day2': [16, 20, None, 11, None, 15, 18, 6, 38, 16],
    'day3': [16, 17, None, 8, None, 11, 14, 6, 37, 16]
}

# Create DataFrame
df = pd.DataFrame(data)

# Fill in missing forecast data for remaining days with simulated values
np.random.seed(42)  # For reproducibility
for i in range(4, 7):
    day_key = f'day{i}'
    df[day_key] = df['today'].apply(lambda x: max(5, min(130, int(x * (0.7 + np.random.randn() * 0.2)))))

# Create date labels for better readability
date_labels = {
    'today': '12 มี.ค.',
    'day1': '13 มี.ค.',
    'day2': '14 มี.ค.',
    'day3': '15 มี.ค.',
    'day4': '16 มี.ค.',
    'day5': '17 มี.ค.',
    'day6': '18 มี.ค.',
}

# Function to determine PM2.5 level category and color
def get_pm25_category(value):
    if value is None:
        return 'ไม่มีข้อมูล', '#cccccc'
    elif value <= 15:
        return 'ดีมาก', '#82cc33'
    elif value <= 25:
        return 'ดี', '#179e4c'
    elif value <= 37.5:
        return 'ปานกลาง', '#fecb38'
    elif value <= 50:
        return 'เริ่มมีผลต่อสุขภาพ', '#f18135'
    elif value <= 90:
        return 'มีผลต่อสุขภาพ', '#e92d25'
    else:
        return 'มีผลต่อสุขภาพมาก', '#7f1b7b'

# Initialize the Dash app
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

# Define CSS styles
colors = {
    'background': '#f8f9fa',
    'text': '#333333',
    'accent': '#6639b6',  # Using purple from the image
    'grid': '#e0e0e0'
}

# Custom CSS
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
                font-family: 'Kanit', 'Sarabun', sans-serif;
                margin: 0;
                background-color: #f8f9fa;
            }
            .header {
                background-color: #6639b6;
                padding: 20px;
                color: white;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .dashboard-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .card {
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                padding: 15px;
                margin-bottom: 20px;
            }
            .card-title {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 15px;
                color: #6639b6;
            }
            .info-box {
                padding: 10px;
                border-radius: 5px;
                background-color: #e9ecef;
                margin-bottom: 15px;
            }
            .tab-content {
                padding-top: 20px;
            }
            .table-container {
                overflow-x: auto;
            }
            .footer {
                text-align: center;
                padding: 20px;
                color: #666;
                font-size: 12px;
            }
            /* PM2.5 level indicators */
            .pm-level {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                color: white;
                font-weight: bold;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>รายงานการพยากรณ์ค่า PM2.5</h1>
            <p>ข้อมูลมลพิษ จังหวัดสุราษฏร์ธานี</p>
        </div>
        <div class="dashboard-container">
            {%app_entry%}
        </div>
        <div class="footer">
            © 2025 ระบบพยากรณ์ค่าฝุ่น PM2.5 จังหวัดสุราษฏร์ธานี
        </div>
        {%config%}
        {%scripts%}
        {%renderer%}
    </body>
</html>
'''

# Layout of the app
app.layout = html.Div([
    # Information box
    html.Div([
        html.Div("ข้อมูลการพยากรณ์ค่าฝุ่น PM2.5 ล่วงหน้า 7 วัน", className="card-title"),
        html.P("แสดงข้อมูลการพยากรณ์ค่าฝุ่น PM2.5 ในพื้นที่จังหวัดสุราษฏร์ธานี แยกตามสถานีตรวจวัด")
    ], className="info-box"),
    
    # Tabs
    dcc.Tabs([
        # Tab 1: Overview
        dcc.Tab(label="ภาพรวม", children=[
            html.Div([
                # Summary card
                html.Div([
                    html.Div("สรุปค่าเฉลี่ย PM2.5 รายวัน", className="card-title"),
                    dcc.Graph(
                        figure=px.line(
                            pd.DataFrame({
                                'วันที่': list(date_labels.values()),
                                'ค่าเฉลี่ย PM2.5': [
                                    df['today'].mean(),
                                    df['day1'].mean(),
                                    df['day2'].mean(),
                                    df['day3'].mean(),
                                    df['day4'].mean(),
                                    df['day5'].mean(),
                                    df['day6'].mean()
                                ]
                            }), 
                            x='วันที่', 
                            y='ค่าเฉลี่ย PM2.5',
                            markers=True,
                            line_shape='spline',
                            color_discrete_sequence=['#6639b6']
                        ).update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            xaxis=dict(showgrid=True, gridcolor=colors['grid']),
                            yaxis=dict(showgrid=True, gridcolor=colors['grid'])
                        )
                    )
                ], className="card"),
                
                # Heatmap card
                html.Div([
                    html.Div("แผนที่ความเข้มข้นตามสถานี", className="card-title"),
                    dcc.Graph(
                        figure=px.imshow(
                            df[['today', 'day1', 'day2', 'day3', 'day4', 'day5', 'day6']].values,
                            labels=dict(x="วันที่", y="สถานีตรวจวัด", color="PM2.5"),
                            x=list(date_labels.values()),
                            y=df['station'],
                            color_continuous_scale=[[0, '#82cc33'], [0.2, '#179e4c'], 
                                                  [0.4, '#fecb38'], [0.6, '#f18135'], 
                                                  [0.8, '#e92d25'], [1, '#7f1b7b']],
                            aspect="auto"
                        ).update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                    )
                ], className="card"),
                
                # Highest stations card
                html.Div([
                    html.Div("สถานีที่มีค่า PM2.5 สูงสุด", className="card-title"),
                    dcc.Graph(
                        figure=px.bar(
                            df.sort_values('today', ascending=False).head(5),
                            x='station',
                            y='today',
                            color='today',
                            color_continuous_scale=[[0, '#82cc33'], [0.2, '#179e4c'], 
                                                  [0.4, '#fecb38'], [0.6, '#f18135'], 
                                                  [0.8, '#e92d25'], [1, '#7f1b7b']],
                            labels={'station': 'สถานี', 'today': 'PM2.5 วันนี้'}
                        ).update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=True, gridcolor=colors['grid'])
                        )
                    )
                ], className="card")
            ], className="tab-content"),
        ]),
        
        # Tab 2: Station Details
        dcc.Tab(label="รายละเอียดรายสถานี", children=[
            html.Div([
                # Station dropdown
                html.Div([
                    html.Div("เลือกสถานีตรวจวัด", className="card-title"),
                    dcc.Dropdown(
                        id='station-dropdown',
                        options=[{'label': station, 'value': station} for station in stations],
                        value=stations[0]
                    )
                ], className="card"),
                
                # Station detail graph
                html.Div([
                    html.Div(id='station-title', className="card-title"),
                    dcc.Graph(id='station-graph')
                ], className="card"),
                
                # PM2.5 categories explanation
                html.Div([
                    html.Div("เกณฑ์คุณภาพอากาศ (PM2.5)", className="card-title"),
                    html.Div([
                        html.Div("0-15 μg/m³: ดีมาก", className="pm-level", style={'backgroundColor': '#82cc33'}),
                        html.Div("16-25 μg/m³: ดี", className="pm-level", style={'backgroundColor': '#179e4c', 'margin-left': '5px'}),
                        html.Div("26-37.5 μg/m³: ปานกลาง", className="pm-level", style={'backgroundColor': '#fecb38', 'margin-left': '5px'}),
                        html.Div("38-50 μg/m³: เริ่มมีผลต่อสุขภาพ", className="pm-level", style={'backgroundColor': '#f18135', 'margin-left': '5px'}),
                        html.Div("51-90 μg/m³: มีผลต่อสุขภาพ", className="pm-level", style={'backgroundColor': '#e92d25', 'margin-left': '5px'}),
                        html.Div(">90 μg/m³: มีผลต่อสุขภาพมาก", className="pm-level", style={'backgroundColor': '#7f1b7b', 'margin-left': '5px'})
                    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px', 'marginTop': '10px'})
                ], className="card")
            ], className="tab-content")
        ]),
        
        # Tab 3: Data Table
        dcc.Tab(label="ตารางข้อมูล", children=[
            html.Div([
                html.Div("ข้อมูลการพยากรณ์ค่า PM2.5 รายสถานี", className="card-title"),
                html.Div([
                    dash_table.DataTable(
                        id='data-table',
                        columns=[
                            {'name': 'สถานี', 'id': 'station'},
                            {'name': date_labels['today'], 'id': 'today'},
                            {'name': date_labels['day1'], 'id': 'day1'},
                            {'name': date_labels['day2'], 'id': 'day2'},
                            {'name': date_labels['day3'], 'id': 'day3'},
                            {'name': date_labels['day4'], 'id': 'day4'},
                            {'name': date_labels['day5'], 'id': 'day5'},
                            {'name': date_labels['day6'], 'id': 'day6'},
                        ],
                        data=df.to_dict('records'),
                        style_header={
                            'backgroundColor': colors['accent'],
                            'color': 'white',
                            'fontWeight': 'bold',
                            'textAlign': 'center'
                        },
                        style_cell={
                            'textAlign': 'center',
                            'padding': '8px'
                        },
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'station'},
                                'textAlign': 'left',
                                'fontWeight': 'bold'
                            },
                            # Color coding for PM2.5 values
                            {
                                'if': {'filter_query': '{today} <= 15'},
                                'backgroundColor': '#e5f7d3',
                                'color': '#179e4c'
                            },
                            {
                                'if': {'filter_query': '{today} > 15 && {today} <= 25'},
                                'backgroundColor': '#d0ecd1',
                                'color': '#179e4c'
                            },
                            {
                                'if': {'filter_query': '{today} > 25 && {today} <= 37.5'},
                                'backgroundColor': '#fff4d1',
                                'color': '#b58a00'
                            },
                            {
                                'if': {'filter_query': '{today} > 37.5 && {today} <= 50'},
                                'backgroundColor': '#ffead1',
                                'color': '#d05600'
                            },
                            {
                                'if': {'filter_query': '{today} > 50 && {today} <= 90'},
                                'backgroundColor': '#ffd5d1',
                                'color': '#9c0000'
                            },
                            {
                                'if': {'filter_query': '{today} > 90'},
                                'backgroundColor': '#ecd2ec',
                                'color': '#7f1b7b'
                            },
                            # Repeat for other days
                            # day1
                            {
                                'if': {'filter_query': '{day1} <= 15'},
                                'backgroundColor': '#e5f7d3',
                                'color': '#179e4c'
                            },
                            # ... similar conditions for all other columns/days
                        ]
                    )
                ], className="table-container"),
            ], className="card tab-content")
        ])
    ])
])

# Callbacks
@app.callback(
    [dash.dependencies.Output('station-title', 'children'),
     dash.dependencies.Output('station-graph', 'figure')],
    [dash.dependencies.Input('station-dropdown', 'value')]
)
def update_station_graph(selected_station):
    # Filter data for selected station
    station_data = df[df['station'] == selected_station].iloc[0]
    
    # Create data for the graph
    graph_data = {
        'วันที่': list(date_labels.values()),
        'PM2.5': [
            station_data['today'],
            station_data['day1'],
            station_data['day2'],
            station_data['day3'],
            station_data['day4'],
            station_data['day5'],
            station_data['day6']
        ]
    }
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=graph_data['วันที่'],
        y=graph_data['PM2.5'],
        marker_color=[
            get_pm25_category(val)[1] for val in graph_data['PM2.5']
        ],
        text=graph_data['PM2.5'],
        textposition='auto'
    ))
    
    # Add line for threshold
    fig.add_shape(
        type="line",
        x0=0,
        y0=37.5,
        x1=len(graph_data['วันที่'])-1,
        y1=37.5,
        line=dict(
            color="#f18135",
            width=2,
            dash="dash",
        )
    )
    
    fig.add_annotation(
        x=len(graph_data['วันที่'])-1,
        y=37.5,
        text="เกณฑ์เริ่มมีผลต่อสุขภาพ",
        showarrow=False,
        yshift=10,
        font=dict(
            size=12,
            color="#f18135"
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"การพยากรณ์ค่า PM2.5 สถานี{selected_station} อีก 7 วันข้างหน้า",
        xaxis_title="วันที่",
        yaxis_title="ค่า PM2.5 (μg/m³)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=colors['grid'])
    )
    
    return f"การพยากรณ์ค่า PM2.5 สถานี {selected_station}", fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)