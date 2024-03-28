from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx
from dash.exceptions import PreventUpdate


import json
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from io import BytesIO, StringIO

df = pd.read_csv("src/data/tcc_ceds_music.csv")

FEATURES = ['release_date', 'genre', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'topic']
TARGET = ['danceability', 'energy']

# all features for the dimensionality reduction + model (temporary)
NUM_FEATURES = ['release_date', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'blues', 'country', 'hip hop', 
                'jazz', 'pop', 'reggae', 'rock', 'feelings', 'music', 'night/time', 'obscene', 'romantic', 'sadness', 
                'violence', 'world/life']

TEST_FEATURES = ['loudness', 'acousticness', 'instrumentalness', 'valence', 'blues', 'country', 'hip hop', 
                'jazz', 'pop', 'reggae', 'rock', 'feelings', 'music', 'night/time', 'obscene', 'romantic', 'sadness', 
                'violence', 'world/life']

ALL_FEATURES = ['loudness', 'acousticness', 'instrumentalness',
                'valence', 'blues', 'country', 'hip hop', 'jazz', 'pop', 'reggae',
                'rock', 'feelings', 'music', 'night/time', 'obscene', 'romantic',
                'sadness', 'violence', 'world/life', 'dance', 'ener', 'x_coor',
                'y_coor', 'artist', 'track', 'preference']

X = df[FEATURES]
target = df[TARGET]

# one hot encode the categorical columns
def one_hot_encode(X, column):
    enc = OneHotEncoder(handle_unknown='ignore')
    columns = sorted(X[column].unique())
    enc_df1 = pd.DataFrame(enc.fit_transform(X[[column]]).toarray(), columns=columns)
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
data['preference'] = 0

# create scatterplot
fig = px.scatter(data, x='x_coor', y='y_coor', 
                                hover_data={'x_coor':False, 
                                            'y_coor':False,
                                            'artist':True,
                                            'track':True,
                                            'release_date':True,
                                            })

fig.update_layout(clickmode='event+select')

# create dashboard with graph and table
app = Dash(__name__)
app.layout = html.Div([
    html.Div(children='My First App with Data Graph'),
    html.Hr(),
    dcc.Graph(id='main-vis', figure=fig),
    dash_table.DataTable(
        id='overview-table',
        page_size=6,
        # columns=[{'name': col, 'id': col} for col in data.columns],
        columns=[{
            'id': 'release_date',
            'name': 'release_date',
            'type': 'numeric'
        }, {
            'id': 'loudness',
            'name': 'loudness',
            'type': 'numeric',
        }, {
            'id': 'acousticness',
            'name': 'acousticness',
            'type': 'numeric',
        }, {
            'id': 'instrumentalness',
            'name': 'instrumentalness',
            'type': 'numeric',
        }, {
            'id': 'valence',
            'name': 'valence',
            'type': 'numeric',
        }, {
            'id': 'blues',
            'name': 'blues',
            'type': 'numeric',
        }, {
            'id': 'country',
            'name': 'country',
            'type': 'numeric',
        }, {
            'id': 'hip hop',
            'name': 'hip hop',
            'type': 'numeric',
        }, {
            'id': 'jazz',
            'name': 'jazz',
            'type': 'numeric',
        }, {
            'id': 'pop',
            'name': 'pop',
            'type': 'numeric',
        }, {
            'id': 'reggae',
            'name': 'reggae',
            'type': 'numeric',
        }, {
            'id': 'rock',
            'name': 'rock',
            'type': 'numeric',
        }, {
            'id': 'feelings',
            'name': 'feelings',
            'type': 'numeric',
        }, {
            'id': 'music',
            'name': 'music',
            'type': 'numeric',
        }, {
            'id': 'night/time',
            'name': 'night/time',
            'type': 'numeric',
        }, {
            'id': 'obscene',
            'name': 'obscene',
            'type': 'numeric',
        }, {
            'id': 'romantic',
            'name': 'romantic',
            'type': 'numeric',
        }, {
            'id': 'sadness',
            'name': 'sadness',
            'type': 'numeric',
        }, {
            'id': 'violence',
            'name': 'violence',
            'type': 'numeric',
        }, {
            'id': 'world/life',
            'name': 'world/life',
            'type': 'numeric',
        }, {
            'id': 'dance',
            'name': 'dance',
            'type': 'numeric',
        }, {
            'id': 'ener',
            'name': 'ener',
            'type': 'numeric',
        }, {
            'id': 'artist',
            'name': 'artist',
            'type': 'text',
        }, {
            'id': 'track',
            'name': 'track',
            'type': 'text',
        }],
        data=data.to_dict('records'),
        editable=True,  
    ),
    dcc.Store(id='data-store'),
    # html.Button('Train model!', id='train-btn', n_clicks=0),
    # html.Div(id='container-button-timestamp'),
])

# TODO: fix zoom: relayoutData?
# TODO: fix lasso/box select
# Callback to update DataTable and visualization based on selections
@app.callback(
    Output('main-vis', 'figure'),
    Output('data-store', 'data'),
    Output('overview-table', 'data'),
    Input('overview-table', 'data'),
    Input('main-vis', 'selectedData'),
)
def update_data(table_data, vis_data):
    if table_data is None:
        # If no data is selected, display the entire DataFrame
        table_df = data
    else:
        # otherwise, display the selected data
        table_df = pd.DataFrame(table_data)

    print(vis_data)

    if vis_data is None:
        # If no data is selected, display the entire DataFrame
        vis_df = data
    else:
        # Extract selected points from the scatterplot
        selected_points = [point['pointIndex'] for point in vis_data['points']]
        
        # Filter DataFrame based on selected points
        vis_df = data.iloc[selected_points]

    # create a boolean mask for all rows that are present in both 'data' and 'table_df', based on TEST_FEATURES
    mask_b_c = data[TEST_FEATURES].isin(table_df[TEST_FEATURES].to_dict(orient='list')).all(axis=1)
    diff = len(mask_b_c) - len(table_df)
    # if data is larger than table_df, we need to append to other mask
    if diff > 0:
        # create a mask that checks whether a values has changed in a row
        merged_df = data.merge(table_df, on=TEST_FEATURES, how='left')
        true_indices = merged_df[~merged_df.isnull().any(axis=1)].index.tolist()
        mask_a = np.full(len(mask_b_c), False)
        mask_a[true_indices] = True
        # combine both masks
        result = (mask_b_c & mask_a)
    # if data is not larger, we can simply create the mask
    else:   
        # create a mask that checks whether the 'release_date' value has changed per row
        mask_a = data[mask_b_c]['release_date'].reset_index(drop=True) != table_df['release_date'].reset_index(drop=True)
        # combine both masks
        result = (mask_b_c & mask_a)

    # get the indices of the rows where both masks are true
    indices_in_df2 = data[result].index.tolist()

    # replace row in 'data' with altered row from 'table_df'
    if len(table_df) == len(data):
        for i in range(0, len(indices_in_df2)):
            data.iloc[indices_in_df2[i]] = table_df.iloc[indices_in_df2[i]].tolist()
    elif len(table_df) < len(data):
        table_df2 = table_df.merge(vis_df, on=ALL_FEATURES, how='left')[['release_date_x'] + ALL_FEATURES].copy().rename(columns={"release_date_x": "release_date"})
        for i in range(0, len(indices_in_df2)):
            data.iloc[indices_in_df2[i]] = table_df2.iloc[i].tolist()


        # repeat the same process for the visualization dataframe
        # create the masks, and replace rows in dataframe
        small_mask_b_c = vis_df[TEST_FEATURES].isin(table_df[TEST_FEATURES].to_dict(orient='list')).all(axis=1)
        small_merged_df = vis_df.merge(table_df, on=TEST_FEATURES, how='left')
        small_true_indices = small_merged_df[~small_merged_df.isnull().any(axis=1)].index.tolist()
        small_false_values = np.full(len(small_mask_b_c), False)
        small_false_values[small_true_indices] = True
        small_result = (small_mask_b_c & small_false_values)
        small_indices_in_df2 = vis_df[small_result].index.tolist()
        for i in range(0, len(small_indices_in_df2)):
            vis_df['release_date'].loc[small_indices_in_df2[i]] = table_df['release_date'].iloc[i].tolist()

    # create new reduction
    new_X_reduced = tsne.fit_transform(data[NUM_FEATURES])
    data['x_coor'], data['y_coor'] = new_X_reduced[:,0], new_X_reduced[:,1]
    
    # create new scatter plot
    fig = px.scatter(data, x='x_coor', y='y_coor',
                    hover_data={'x_coor': False,
                                'y_coor': False,
                                'artist': True,
                                'track': True,
                                'release_date': True,
                                })
    fig.update_layout(clickmode='event+select')
    fig.update_traces(selectedpoints=vis_df.index)

    return fig, json.dumps(data.to_dict("index")), vis_df.to_dict('records')

# Callback to update DataTable based on changed in the scatterplot (S2T)
# @app.callback(
#     Output('overview-table', 'data'),
#     [Input('main-vis', 'selectedData')]
# )
# def update_datatable(selected_data):
#     print('Activated S2T')
#     if selected_data is None:
#         # If no data is selected, display the entire DataFrame
#         filtered_df = data
#     else:
#         # Extract selected points from the scatterplot
#         selected_points = [point['pointIndex'] for point in selected_data['points']]
        
#         # Filter DataFrame based on selected points
#         filtered_df = data.iloc[selected_points]

#     print('data:', filtered_df)

#     return filtered_df.to_dict('records')

# Callback to update scatterplot based on changed in the DataTable (T2S)
# @app.callback(
#     Output('main-vis', 'figure'),
#     Output('data-store', 'data'),
#     Input('overview-table', 'data'),
# )
# def update_scatterplot(table_data): 
#     print('Entered T2S')
#     if table_data is None:
#         # If no data is selected, display the entire DataFrame
#         table_df = data
#     else:
#         # otherwise, display the selected data
#         table_df = pd.DataFrame(table_data)

#     # create new reduction
#     new_X_reduced = tsne.fit_transform(table_df[NUM_FEATURES])
#     table_df['x_coor'], table_df['y_coor'] = new_X_reduced[:,0], new_X_reduced[:,1]
    
#     # create new scatter plot
#     fig = px.scatter(table_df, x='x_coor', y='y_coor',
#                     hover_data={'x_coor': False,
#                                 'y_coor': False,
#                                 'artist': True,
#                                 'track': True,
#                                 'release_date': True,
#                                 })
#     fig.update_layout(clickmode='event+select')

#     return fig, json.dumps(data.to_dict("index"))

# @callback(
#     Output('container-button-timestamp', 'children'),
#     Input('train-btn', 'n_clicks'),
#     Input('data-store', 'data'),
# )
# def displayClick(btn1, data):
#     df = pd.read_json(StringIO(data)).transpose()
#     print('BUTTON DATA', df.head())
#     df.to_csv('convert3.csv')

#     msg = 'Start labeling!'
#     if "train-btn" == ctx.triggered_id:
#         if np.count_nonzero(df['preference']) < 3:
#             msg = "Label at least 3 instances before training the model"
#         else:
#             msg = "Training model..."
#             X_train = df[(df[['preference']] != 0).all(axis=1)][NUM_FEATURES]
#             y_train = df[(df[['preference']] != 0).all(axis=1)]['dance'].astype('int')
#             X_test = df[(df[['preference']] == 0).all(axis=1)][NUM_FEATURES]
#             y_test = df[(df[['preference']] == 0).all(axis=1)]['dance'].astype('int')

#             rfc = RandomForestClassifier(n_estimators=100, random_state=0)
#             rfc.fit(X_train, y_train)
#             y_pred = rfc.predict(X_test)

#             print(sum(y_pred == y_test) / len(y_test))
#     return html.Div(msg)

if __name__ == '__main__':
    app.run(debug=True)