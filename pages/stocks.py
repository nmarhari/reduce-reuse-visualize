from dash import Dash, dcc, html, Input, Output
import plotly.express as px

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

df = px.data.stocks()

#app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([

    html.H2('Dashboard'),

    html.P('This graph shows ...'),

    # html.Div(id='dropdown-container', children=[
    #     dcc.Dropdown(id="ticker",
    #              options=["AMZN", "FB", "NFLX"],
    #              clearable = False,
    #              value = "FB",
    #              #style={'width': "40%"}
    #              ),
    # ]),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='time-series-chart'),
    
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