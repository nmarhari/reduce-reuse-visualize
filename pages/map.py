import dash
from dash import Dash, dcc, html, Input, Output, callback, State, ctx
import plotly.express as express
import plotly.graph_objects as graphObjects
import pandas

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
df = pandas.read_csv("./datasets/intro_bees.csv")

df = df.groupby(['State', 'ANSI', 'Affected by', 'Year', 'state_code'])[['Pct of Colonies Impacted']].mean()
df.reset_index(inplace=True)
# print(df[:5])

# app
#app = Dash(__name__)


stats_description = df.describe().round(3).reset_index()
eda_table = dbc.Table.from_dataframe(stats_description, striped=True, bordered=True, hover=True, id='EDA-bees')


dash.register_page(__name__, suppress_callback_exceptions=True)

layout = html.Div(id='map-container', children=[

    html.H2('Dashboard'),

    html.P('This graph shows the number of bee colonies impacted by disease, humans, and other external factors around the United States.'),

    # EDA Table
    html.Div([
        html.H3('Exploratory Data Analysis Statistics'),
        dbc.Table.from_dataframe(stats_description, striped=True, bordered=True, hover=True, id='EDA-bees')
    ]),

    html.Div(id='dropdown-container', children=[
        dcc.Dropdown(id="slct_year",
                 options=[
                     {"label": "2015", "value": 2015},
                     {"label": "2016", "value": 2016},
                     {"label": "2017", "value": 2017},
                     {"label": "2018", "value": 2018}],
                 multi=False,
                 value=2015,
                 #style={'width': "40%"}
                 ),
    ]),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='my_bee_map', figure={}, style={'height:': '60vh'}),
    
    dbc.Row([
            dbc.Col(dbc.Button(id='btn', children='About this graph', className='my-2'), width=1)
        ],),
    dbc.Row([
            dbc.Col(dcc.Loading(html.Div(style={'margin':'0 20vw'},id='content', children=''), fullscreen=False), width=6)
    ],),

    dbc.Row([
        dbc.Col(id='question-row', children=[
            dcc.Markdown(
                "#### Need any further clarification? Ask our AI!\n",
                style={"whiteSpace": "pre", 'margin': '.5vh .5vh'},
            ),
            dbc.Input(
                id="input-id2",
                placeholder="Type your question...",
                type="text",
            ),
            dbc.Col(style={'margin': '.5vh .5vh'}, children=[
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
            dcc.Loading(children=html.P(style={'margin':'0 20vw', 'margin-bottom':'2vh'},id="output-id2")),
        ],
        width=10,
        ),
    ]),
])

# Year Chosen Dropdown
@callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='my_bee_map', component_property='figure')],
    [Input(component_id='slct_year', component_property='value')]
)
def update_graph(option_slctd):
    # print(option_slctd)
    # print(type(option_slctd))

    container = "The year chosen by user was: {}".format(option_slctd)

    dff = df.copy()
    dff = dff[dff["Year"] == option_slctd]
    dff = dff[dff["Affected by"] == "Varroa_mites"]

    # Plotly Express
    fig = express.choropleth(
        data_frame=dff,
        locationmode='USA-states',
        locations='state_code',
        scope="usa",
        color='Pct of Colonies Impacted',
        hover_data=['State', 'Pct of Colonies Impacted'],
        color_continuous_scale=express.colors.sequential.YlOrRd,
        labels={'Pct of Colonies Impacted': '% of Bee Colonies Impacted'}
    )
    return container, fig

# AI Insights Button
@callback(
    Output('content','children'),
    Input('btn','n_clicks'),
    State('my_bee_map','figure'),
    prevent_initial_call=True
)
def graph_insights(_, fig):
    fig_object = go.Figure(fig)
    fig_object.write_image(f"images/fig{_}.png")
    time.sleep(1)

    chat = ChatOpenAI(model="gpt-4-turbo", max_tokens=256)
    image_path = f"images/fig{_}.png"
    base64_image = encode_image(image_path)
    result = chat.invoke(
        [
            # Limitations of the model -- https://platform.openai.com/docs/guides/vision/limitations
            # SystemMessage(
            #     content="You are an expert in data visualization that reads images of graphs and describes the data trends in those images. "
            #             "The graphs you will read are line charts that have multiple lines in them. Please pay careful attention to the "
            #             "legend color of each line and match them to the line color in the graph. The legend colors must match the line colors "
            #             "in the graph correctly."
            # ),
            HumanMessage(
                content=[
                    {"type": "text", "text": "What data insight can we get from this graph? Limit your response to 1000 characters of plain text."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "auto",  # https://platform.openai.com/docs/guides/vision/low-or-high-fidelity-image-understanding
                        },
                    },
                ]
            )
        ]
    )
    return result.content

# AI Question Box
@callback(
    Output("output-id2", "children"),
    [Input("btn2", "n_clicks"), Input("btn2-reset", "n_clicks")],
    State('my_bee_map', 'figure'),
    State("input-id2", "value"),
    prevent_initial_call=True,
)
def data_insights(_, _reset, fig, value):
    button_clicked = ctx.triggered_id

    if button_clicked == "btn2":
        fig_object = go.Figure(fig)
        fig_object.write_image(f"images/fig{_}.png")
        time.sleep(1)

        chat = ChatOpenAI(model="gpt-4-turbo", max_tokens=256)
        image_path = f"images/fig{_}.png"
        base64_image = encode_image(image_path)
        
        # dataset = dfList[active_page - 1]
        # agent = create_pandas_dataframe_agent(chat, dataset, verbose=True)
        if value is None:
            resp_output = "No question provided."
        else:
            question = f"{value}"
            question+=" Limit your response to 1000 characters of plain text."
            print(value)
            print(question)
            try:
                result = chat.invoke ([HumanMessage(
                    content=[
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "auto",  # https://platform.openai.com/docs/guides/vision/low-or-high-fidelity-image-understanding
                            },
                        },
                    ]
                )])
                print(result.content)
                # response = agent.invoke(question)
                # resp_output = f"{response['output']}"
                resp_output = result.content
                value = ''
            except:
                resp_output = "Sorry, your question is out of context"
        return resp_output
    elif button_clicked == "btn2-reset":
        return ""

# --
#if __name__ == '__main__':
  #  run_server(debug=True)