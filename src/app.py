from dash import Dash, html, dash_table, dcc, callback, Output, Input
from dash.exceptions import PreventUpdate

import json
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

df = pd.read_csv("src/data/tcc_ceds_music.csv")

FEATURES = ['release_date', 'genre', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'topic']
TARGET = ['danceability', 'energy']

X = df[FEATURES]
target = df[TARGET]

# one hot encode the categorical columns
def one_hot_encode(X, column):
    enc = OneHotEncoder(handle_unknown='ignore')
    # passing bridge-types-cat column (label encoded values of bridge_types)
    columns = sorted(X[column].unique())
    enc_df1 = pd.DataFrame(enc.fit_transform(X[[column]]).toarray(), columns=columns)
    # merge with main df bridge_df on key values
    X = X.join(enc_df1)
    X = X.drop([column], axis=1)
    return X

X = one_hot_encode(X, 'genre')
X = one_hot_encode(X, 'topic')

# put target variables in bins
bins = [-0.1, 0.25, 0.5, 0.75, 1]
labels = [1, 2, 3, 4]
y = target.copy()
y['dc'] = pd.cut(target['danceability'], bins=bins, labels=labels)
y['en'] = pd.cut(target['energy'], bins=bins, labels=labels)

# create dimensionality reduction
tsne = PCA(n_components = 2, random_state = 42)
X_reduced = tsne.fit_transform(X)

# create new dataframe
data = X.copy()
data['dance'], data['ener'] = y['dc'].copy(), y['en'].copy()
data['x_coor'], data['y_coor'] = X_reduced[:,0], X_reduced[:,1]
data['artist'], data['track'] = df[['artist_name']], df[['track_name']]

fig = px.scatter(data, x='x_coor', y='y_coor', 
                                hover_data={'x_coor':False, 
                                            'y_coor':False,
                                            'artist':True,
                                            'track':True,
                                            'release_date':True,
                                            })

fig.update_layout(clickmode='event+select')

app = Dash(__name__)

app.layout = html.Div([
    html.Div(children='My First App with Data Graph'),
    html.Hr(),
    dcc.Graph(id='main-vis', figure=fig),
    dash_table.DataTable(
        id='overview-table',
        page_size=6,
        columns=[{'name': col, 'id': col} for col in data.columns],
        # columns=[{
        #     'id': 'release_date',
        #     'name': 'release_date',
        #     'type': 'numeric'
        # }, {
        #     'id': 'loudness',
        #     'name': 'loudness',
        #     'type': 'numeric',
        # }, {
        #     'id': 'acousticness',
        #     'name': 'acousticness',
        #     'type': 'numeric',
        # }, {
        #     'id': 'instrumentalness',
        #     'name': 'instrumentalness',
        #     'type': 'numeric',
        # }, {
        #     'id': 'valence',
        #     'name': 'valence',
        #     'type': 'numeric',
        # }, {
        #     'id': 'blues',
        #     'name': 'blues',
        #     'type': 'numeric',
        # }, {
        #     'id': 'country',
        #     'name': 'country',
        #     'type': 'numeric',
        # }, {
        #     'id': 'hip hop',
        #     'name': 'hip hop',
        #     'type': 'numeric',
        # }, {
        #     'id': 'jazz',
        #     'name': 'jazz',
        #     'type': 'numeric',
        # }, {
        #     'id': 'pop',
        #     'name': 'pop',
        #     'type': 'numeric',
        # }, {
        #     'id': 'reggae',
        #     'name': 'reggae',
        #     'type': 'numeric',
        # }, {
        #     'id': 'rock',
        #     'name': 'rock',
        #     'type': 'numeric',
        # }, {
        #     'id': 'feelings',
        #     'name': 'feelings',
        #     'type': 'numeric',
        # }, {
        #     'id': 'music',
        #     'name': 'music',
        #     'type': 'numeric',
        # }, {
        #     'id': 'night/time',
        #     'name': 'night/time',
        #     'type': 'numeric',
        # }, {
        #     'id': 'obscene',
        #     'name': 'obscene',
        #     'type': 'numeric',
        # }, {
        #     'id': 'romantic',
        #     'name': 'romantic',
        #     'type': 'numeric',
        # }, {
        #     'id': 'sadness',
        #     'name': 'sadness',
        #     'type': 'numeric',
        # }, {
        #     'id': 'violence',
        #     'name': 'violence',
        #     'type': 'numeric',
        # }, {
        #     'id': 'world/life',
        #     'name': 'world/life',
        #     'type': 'numeric',
        # }, {
        #     'id': 'dance',
        #     'name': 'dance',
        #     'type': 'numeric',
        # }, {
        #     'id': 'ener',
        #     'name': 'ener',
        #     'type': 'numeric',
        # }, {
        #     'id': 'artist',
        #     'name': 'artist',
        #     'type': 'text',
        # }, {
        #     'id': 'track',
        #     'name': 'track',
        #     'type': 'text',
        # }],
        data=data.to_dict('records'),
        editable=True,  
    ),
])

# Callback to update DataTable based on scatterplot selection
# @app.callback(
#     Output('overview-table', 'data'),
#     [Input('main-vis', 'selectedData')]
# )
# def update_datatable(selected_data):
#     if selected_data is None:
#         # If no data is selected, display the entire DataFrame
#         filtered_df = data
#     # elif not selected_data:   
#     #     # If no data is selected, display the entire DataFrame
#     #     filtered_df = data
#     else:
#         # Extract selected points from the scatterplot
#         selected_points = [point['pointIndex'] for point in selected_data['points']]
        
#         # Filter DataFrame based on selected points
#         filtered_df = data.iloc[selected_points]

#     return filtered_df.to_dict('records')

# Callback to update DataTable based on scatterplot selection
@app.callback(
    Output('main-vis', 'figure'),
    [Input('overview-table', 'data')]
)
def update_scatterplot(selected_data):
    if selected_data is None:
        # If no data is selected, display the entire DataFrame
        updated_data = data
    # elif not selected_data:   
    #     # If no data is selected, display the entire DataFrame
    #     updated_data = data
    else:
        updated_data = pd.DataFrame(selected_data)

    # print('len sel', updated_data)
    # print('len data', data)
    
    # with open('convert.txt', 'w') as convert_file: 
    #     convert_file.write(json.dumps(selected_data))

    # data.to_csv('convert2.txt', index=False)

    # mask = data.eq(selected_data[0]).all(axis=1)
    # indices = X.index[mask].tolist()

    # print('mask', mask)
    # print(indices[0])
    # print(indices)
        
    # unequal_rows = data[data['release_date'].ne(updated_data['release_date'])]
    # indices = unequal_rows.index

    # if not indices.empty:
    #     if len(updated_data) < len(indices):
    #         for i in range(0, len(updated_data)):
    #             data.iloc[indices[i]] = updated_data.iloc[i]
    #     elif len(updated_data) > len(indices):
    #         for i in range(0, len(updated_data)):
    #             data.iloc[i] = updated_data.iloc[i]

    # print('data', data.head())

    # print('selected data: ', selected_data)
    # print('updated data: ', updated_data)

    NEW_FEATURES = ['release_date', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'blues', 'country', 'hip hop', 
                    'jazz', 'pop', 'reggae', 'rock', 'feelings', 'music', 'night/time', 'obscene', 'romantic', 'sadness', 
                    'violence', 'world/life']

    # new_X_reduced = tsne.fit_transform(updated_data.drop(['artist', 'track'], axis=1))
    new_X_reduced = tsne.fit_transform(updated_data[NEW_FEATURES])
    updated_data['x_coor'], updated_data['y_coor'] = new_X_reduced[:,0], new_X_reduced[:,1]
    
    fig = px.scatter(updated_data, x='x_coor', y='y_coor',
                    hover_data={'x_coor': False,
                                'y_coor': False,
                                'artist': True,
                                'track': True,
                                'release_date': True,
                                })
    
    # new_X_reduced = tsne.fit_transform(data[NEW_FEATURES])
    # data['x_coor'], data['y_coor'] = new_X_reduced[:,0], new_X_reduced[:,1]
    
    # fig = px.scatter(data, x='x_coor', y='y_coor',
    #                 hover_data={'x_coor': False,
    #                             'y_coor': False,
    #                             'artist': True,
    #                             'track': True,
    #                             'release_date': True,
    #                             })
    
    # fig.update_layout(clickmode='event+select')
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)