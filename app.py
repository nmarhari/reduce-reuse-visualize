import dash
from dash import Dash, html
# pip install -r requirements.txt

app = Dash(__name__, use_pages=True)

app.layout = html.Div([

    html.H1('Reduce, Reuse, Visualize'),
    dash.page_container
])

if __name__ == '__main__':
    app.run(debug=True)