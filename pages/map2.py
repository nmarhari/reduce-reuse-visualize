import dash
from dash import Dash, dcc, html, Input, Output, callback, State, ctx
import plotly.express as px
import plotly.graph_objects as graphObjects
import pandas as pd

#insights
import dash_bootstrap_components as dbc
import time
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import base64

# function for decoding graph image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
#df = pandas.read_csv("<file>")
df = pd.read_csv("./datasets/Electric_Vehicle_Population_Data.csv")

df = df.dropna(subset=['Vehicle Location'])  # Drop rows where 'Vehicle Location' is NaN
#df['Coordinates'] = df['Vehicle Location'].apply(lambda x: x.strip('POINT (').strip(')').split() if isinstance(x, str) else [None, None])
df['Longitude'] = df['Vehicle Location'].apply(lambda x: float(x[7:-1].split()[0]) if isinstance(x, str) and x.startswith('POINT (') else None)
df['Latitude'] = df['Vehicle Location'].apply(lambda x: float(x[7:-1].split()[1]) if isinstance(x, str) and x.startswith('POINT (') else None)


dash.register_page(__name__, suppress_callback_exceptions=True)

layout = html.Div([

    html.H2('Dashboard'),

    html.P('This graph shows '),

    html.Div(id='dropdown-container', children=[
        dcc.Dropdown(id="type-dropdown",
                 options=[
                     {"label": i, "value": i} for i in df['Electric Vehicle Type'].unique()],
                 value='Battery Electric Vehicle (BEV)',
                 clearable=False,
                 #style={'width': "40%"}
                 ),
    ]),

    html.Div(id='output_container2', children=[]),
    html.Br(),

    dcc.Graph(id='map-graph', figure={}),
    
    dbc.Row([
            dbc.Col(dbc.Button(id='btn', children='Insights', className='my-2'), width=1)
        ],),
    dbc.Row([
            dbc.Col(dbc.Spinner(html.Div(id='content', children=''), fullscreen=False), width=6)
    ],),

    dbc.Row([
        dbc.Col([
            dcc.Markdown(
                "#### Need any further clarification? Ask our AI!\n",
                style={"textAlign": "left", "whiteSpace": "pre"},
            ),
            dbc.Input(
                id="input-id2",
                placeholder="Type your question...",
                type="text",
            ),
            dbc.Col([
                dbc.Button(
                    id="btn2",
                    children="Get Insights",
                    className="m-3",
                ),
                dbc.Button(
                    id="btn2-reset",
                    children="Reset",
                    className="m-3",
                ),
            ],
            # width=12,
            ),
            html.Br(),
            dcc.Loading(children=html.P(id="output-id2")),
        ],
        width=10,
        ),
    ]),
])

@callback(
    [Output(component_id='output_container2', component_property='children'),
     Output(component_id='map-graph', component_property='figure')],
    [Input(component_id='type-dropdown', component_property='value')]
)
def update_map(option_slctd):
    container = "The option selected by the user was: {}".format(option_slctd)

    filtered_df = df[df['Electric Vehicle Type'] == option_slctd]
    # fig = px.scatter_geo(filtered_df,
    #                      lat='Latitude',
    #                      lon='Longitude',
    #                      hover_name='Model',
    #                      size='Electric Range',
    #                      projection='albers usa',
    #                      title=f'Distribution of {option_slctd}')
    # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig = go.Scattergeo(
        data_frame=filtered_df,
        lat= df['Latitude'],
        lon= df['Longitude'],
        hover_name= 'Model',
        title=f'Distribution of {option_slctd}',
        size = 'Electric Range'
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return container, fig