import dash
from dash import html

dash.register_page(__name__, path='/')

layout = html.Div(className="home-header", children=[
    #
    # html.Div([
    #     html.Nav([
    #         html.A(href="./", children=(html.I(className="fa-solid fa-bolt", id="logo"))),
    #         html.Div(className="navlinks", children=[
    #             html.Li([
    #                 html.A(href="./", children=["Home"])
    #             ])
    #         ])
    #     ])     
    # ]),
    # 
    html.Div(className="text-box", children=[
        html.P("Use any of the navigation links to continue."),
        html.Div(className="home-links", children=[
            html.Ul([
                html.Li([
                    html.A(href="./map", className="home-button", children="Map")
                ]),
                html.Li([
                    html.A(href="./map", className="home-button", children="Map")
                ]),
                html.Li([
                    html.A(href="./map", className="home-button", children="Map")
                ]),
                html.Li([
                    html.A(href="./map", className="home-button", children="Map")
                ])
            ])
        ])
    ])
    
])