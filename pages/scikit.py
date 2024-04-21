import dash
from dash import Dash, dcc, html, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_ag_grid as dag
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import time
import base64

# data file and cleaning
df = pd.read_csv('./datasets/penguins.csv')
for feat in df.select_dtypes('number').columns:
    df[feat] = df[feat].fillna(df[feat].median())
    df['sex'] = df['sex'].fillna(df['sex'].mode()[0])
cat_feats = ['sex', 'species']
df = pd.get_dummies(df, columns=cat_feats)
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
       'body_mass_g', 'sex_Female', 'sex_Male', 'species_Adelie',
       'species_Chinstrap', 'species_Gentoo']
target = ['island']
X = df[features]
y = df[target]
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=.3, random_state=1)
knn = KNeighborsClassifier().fit(X_tr, y_tr)
clusters = np.arange(2,15)
train_scores = []
test_scores = []
for cluster in clusters:
    knn = KNeighborsClassifier(algorithm='auto', n_neighbors=cluster).fit(X_tr, y_tr)
    train_scores.append(knn.score(X_tr, y_tr))
    test_scores.append(knn.score(X_ts, y_ts))
dict_scores  = {'train score':train_scores,
                'test score':test_scores}
df_scores = pd.DataFrame(dict_scores, index=clusters)
clusters[np.argmax(test_scores)]
bag = BaggingClassifier(estimator=KNeighborsClassifier(n_neighbors=5),
                        random_state=42).fit(X_tr, y_tr)

#visualization
y_pred = pd.Series(bag.predict(X_ts))
test = pd.concat([X_ts.reset_index(), y_pred], axis=1)
conf_matrix = confusion_matrix(y_ts, y_pred) #
features = ['bill_length_mm','flipper_length_mm', 'body_mass_g']
classes = test[0].value_counts().index

df_centroids = pd.DataFrame()
for c in classes:
    dict_centroids = {}
    for feat in features:
        mean = test[test[0]==c][feat].mean()
        dict_centroids[feat] = mean
    dict_centroids['predict class'] = c
    centroids = pd.DataFrame(dict_centroids, index=[0])
    df_centroids = pd.concat([df_centroids, centroids], axis=0)
colors = ['blue', 'red', 'green']
fig = go.Figure()

for i, label in enumerate(set(classes)):
    fig.add_trace(go.Scatter3d(
        x=test[test[0]==label]['bill_length_mm'],
        y=test[test[0]==label]['flipper_length_mm'],
        z=test[test[0]==label]['body_mass_g'],
        mode='markers',
        name=f"{label}<br>Island<br>Sample",
        marker={'symbol': 'circle'},
        hovertemplate = 
        'Bill: %{x}' +
        '<br>Flipper: %{y}' +
        '<br>Mass: %{z}',
    )
)

for i, label in enumerate(set(classes)):
    # Add centroids as separate points
    fig.add_trace(go.Scatter3d(
        x=df_centroids[df_centroids['predict class']==label]['bill_length_mm'],
        y=df_centroids[df_centroids['predict class']==label]['flipper_length_mm'],
        z=df_centroids[df_centroids['predict class']==label]['body_mass_g'],
        mode='markers',
        name=f"Centroid<br>{label}",
        marker={'symbol': 'circle-open', 'color': colors[i],
                'size':35},
        hovertemplate = 
        'Bill: %{x}' +
        '<br>Flipper: %{y}' +
        '<br>Mass: %{z}',
    )
)
      
fig.update_layout(
    scene=dict(
        xaxis_title='Bill Length (mm)',
        yaxis_title='Flipper Length (mm)',
        zaxis_title='Body Mass (g)',
    ),
    title='Penguin Measurements with Centroids',
)

stats_description = df.describe().reset_index().round(3)
eda_table = dbc.Table.from_dataframe(stats_description, striped=True, bordered=True, hover=True)

dash.register_page(__name__, suppress_callback_exceptions=True)

layout = html.Div(id='scikit-container', children=[

    html.H2('Dashboard'),

    html.P('This graph shows measurements of penguin\'s features separated by which island they reside on.'),

    dcc.Graph(figure = fig, id='penguin-scatter', style={'height:': '60vh'}),
    # EDA Table
    html.Div([
        html.H3('Exploratory Data Analysis Statistics'),
        dbc.Table.from_dataframe(stats_description, striped=True, bordered=True, hover=True, id='EDA-penguins')
    ]),
    dbc.Row([
            dbc.Col(dbc.Button(id='btn', children='About this graph', className='my-2'), width=1)
        ],),
    dbc.Row([
            dbc.Col(dcc.Loading(html.Div(style={'margin':'0 20vw'},id='insight-scikit', children=''), fullscreen=False), width=6)
    ],),

    dbc.Row(id='question-row', children=[
        dbc.Col([
            dcc.Markdown(
                "#### Ask a question about this graph:\n",
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
            dcc.Loading(children=html.P(style={'margin':'0 20vw', 'margin-bottom':'2vh'},id="ask-scikit-output")),
        ],
        width=10,
        ),
    ]),
])

# function for decoding graph image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# AI Insights Button
@callback(
    Output('insight-scikit','children'),
    Input('btn','n_clicks'),
    State('penguin-scatter','figure'),
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
                    {"type": "text", "text": "What data insight can we get from this graph? Limit your response to 1000 characters of plain text (No Markdown)."},
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
    Output("ask-scikit-output", "children"),
    [Input("btn2", "n_clicks"), Input("btn2-reset", "n_clicks")],
    State('penguin-scatter', 'figure'),
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
            question+=" Limit your response to 1000 characters of plain text (No Markdown)."
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