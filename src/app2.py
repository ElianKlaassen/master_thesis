from dash import Dash, html, dash_table, dcc, Output, Input
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

df = pd.read_csv("src/data/tcc_ceds_music.csv")

X = df[['release_date', 'genre', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'topic']]
y = df[['danceability', 'energy']]

enc = OneHotEncoder(handle_unknown='ignore')
enc_df1 = pd.DataFrame(enc.fit_transform(X[['genre']]).toarray(), columns=['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7'])
X = X.join(enc_df1)

enc = OneHotEncoder(handle_unknown='ignore')
enc_df2 = pd.DataFrame(enc.fit_transform(X[['topic']]).toarray(), columns=['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8'])
X = X.join(enc_df2)

X = X.drop(['genre', 'topic'], axis=1)

bins = np.array([-0.1, 0.25, 0.5, 0.75, 1])
y["dc"] = pd.cut(y['danceability'], bins, labels=[1, 2, 3, 4])
y["en"] = pd.cut(y['energy'], bins, labels=[1, 2, 3, 4])

tsne = PCA(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X)

data = X.copy()
data['dance'] = y['dc'].copy()
data['ener'] = y['en'].copy()
data['x_coor'] = X_reduced[:, 0]
data['y_coor'] = X_reduced[:, 1]
data['artist'] = df[['artist_name']]
data['track'] = df[['track_name']]

fig = px.scatter(data, x='x_coor', y='y_coor',
                 hover_data={'x_coor': False,
                             'y_coor': False,
                             'artist': True,
                             'track': True,
                             'release_date': True,
                             })

fig.update_layout(clickmode='event+select')

app = Dash(__name__)

app.layout = html.Div([
    html.Div(children='My First App with Data Graph'),
    html.Hr(),

    html.Div([
        dcc.Graph(id='main-vis', figure=fig),
        dash_table.DataTable(data=data.to_dict('records'), page_size=6, id='overview-table'),
    ]),

])

# Callback to update scatterplot based on DataTable changes
@app.callback(
    Output('main-vis', 'figure'),
    [Input('overview-table', 'data')]
)
def update_scatterplot(data):
    updated_data = pd.DataFrame(data)
    fig = px.scatter(updated_data, x='x_coor', y='y_coor',
                     hover_data={'x_coor': False,
                                 'y_coor': False,
                                 'artist': True,
                                 'track': True,
                                 'release_date': True,
                                 })

    fig.update_layout(clickmode='event+select')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
