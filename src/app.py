from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx

import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from copy import deepcopy
import random

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

from modAL.models import ActiveLearner, Committee, CommitteeRegressor
from modAL.disagreement import vote_entropy_sampling
from modAL.disagreement import max_std_sampling
from modAL.uncertainty import margin_sampling, classifier_uncertainty, classifier_entropy, classifier_margin

df = pd.read_csv("src/data/tcc_ceds_music.csv")

scaler = MinMaxScaler()
norm_date = scaler.fit_transform(np.array(df['release_date']).reshape(-1, 1))
df['release_date'] = norm_date

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

CHANGE_FEATURES = ['release_date', 'preference']

X = df[FEATURES]
Y = df[TARGET]

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

# create dimensionality reduction
tsne = PCA(n_components = 2, random_state = 42)
X_reduced = tsne.fit_transform(X)

# create new dataframe
data = X.copy()
data['x_coor'], data['y_coor'] = X_reduced[:,0], X_reduced[:,1]
data['artist'], data['track'] = df[['artist_name']], df[['track_name']]
data['preference'] = 0

# create training dataset
train = pd.concat([X, Y], axis=1)

# generate the pool
X_pool = deepcopy(train[NUM_FEATURES].to_numpy())
y_pool1 = deepcopy(train['danceability'].to_numpy())
y_pool2 = deepcopy(train['energy'].to_numpy())
y_pool = [y_pool1, y_pool2]

# initializing Committee members
n_comm = 2
comm_list = list()
query_idx = []
# create a committee for each target
for comm_idx in range(n_comm):
    n_members = 3
    learner_list = list()
    # add members to each committee
    for member_idx in range(n_members):
        # initial training data
        n_initial = 5
        train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
        X_train = X_pool[train_idx]
        y_train = y_pool[comm_idx][train_idx]

        # creating a reduced copy of the data with the known instances removed
        X_pool = np.delete(X_pool, train_idx, axis=0)
        for cidx in range(n_comm):
            y_pool[cidx] = np.delete(y_pool[cidx], train_idx, axis=0)


        # initializing learner
        learner = ActiveLearner(
            estimator=Ridge(alpha=1.0),
            X_training=X_train, y_training=y_train
        )
        learner_list.append(learner)

    # initializing committee    
    committee = CommitteeRegressor(
        learner_list=learner_list,
        query_strategy=max_std_sampling
    )    
    comm_list.append(committee)

# return the Mean Squared Error (MSE)
def get_mse(committee, X, Y_true):
    MSE_dance = 0
    Y_pred = committee.predict(X)
    MSE_dance = MSE_dance + mean_squared_error(Y_pred, Y_true) 
    return MSE_dance

# create a plot for the MSE over time
performance_history = [[1] for i in range(n_comm)]
for comm_idx in range(n_comm):
    unqueried_score = get_mse(comm_list[comm_idx], X_pool, y_pool[comm_idx])
    performance_history[comm_idx] = [unqueried_score]

df_perf = pd.DataFrame(np.transpose(performance_history), columns=['danceability', 'energy'])
fig2 = px.line(df_perf)

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
        columns = [{'id': name, 'name': name, 'type': 'numeric' if name != 'artist' and name != 'track' else 'text'} for name in data.columns],
        data=data.to_dict('records'),
        editable=True,  
    ),
    dcc.Store(id='data-store'),
    html.Button('Add 10 extra instances and train the model!', id='train-btn', n_clicks=0),
    html.Div(id='container-button-timestamp'),
    dcc.Graph(id='acc-vis', figure=fig2),
    dcc.Store(id='x_pool-store', data=X_pool),
    dcc.Store(id='y_pool-store', data=y_pool),
    dcc.Store(id='query-store', data=query_idx),
])

# TODO: fix lasso/box select
# Callback to update DataTable and visualization based on selections
@app.callback(
    Output('main-vis', 'figure'),
    Output('data-store', 'data'),
    Output('overview-table', 'data'),
    Input('overview-table', 'data'),
    Input('main-vis', 'selectedData'),
    Input('main-vis', 'relayoutData'),
    Input('query-store', 'data'),
)
def update_data(table_data, vis_data, relayout_data, query_idx):
    triggered_id = ctx.triggered_prop_ids
    # print('trigger', triggered_id)

    if table_data is None:
        # If no data is selected, display the entire DataFrame
        table_df = data
    else:
        # otherwise, display the selected data
        table_df = pd.DataFrame(table_data)

    # convert int columns to float
    for column in table_df.columns:
        if table_df[column].dtype == 'int64':
            table_df[column] = table_df[column].astype(float)

    if vis_data is None:
        # If no data is selected, display the entire DataFrame
        vis_df = data
    else:
        # Extract selected points from the scatterplot
        selected_points = [point['pointIndex'] for point in vis_data['points']]
        
        # Filter DataFrame based on selected points
        vis_df = data.iloc[selected_points]

    if triggered_id == {'overview-table.data': 'overview-table'}:
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
            mask_a = data[mask_b_c][CHANGE_FEATURES].reset_index(drop=True) != table_df[CHANGE_FEATURES].reset_index(drop=True)

            if type(CHANGE_FEATURES) != str:
                mask_a = mask_a.any(axis=1)

            # combine both masks
            result = (mask_b_c & mask_a)

        # get the indices of the rows where both masks are true
        indices_in_df2 = data[result].index.tolist()

        # replace row in 'data' with altered row from 'table_df'
        if len(table_df) == len(data):
            data.loc[data.index[indices_in_df2]] = table_df.loc[table_df.index[indices_in_df2]].values
            # data.iloc[indices_in_df2] = table_df.iloc[indices_in_df2].values
        elif len(table_df) < len(data):
            data.iloc[indices_in_df2] = table_df.merge(vis_df, on=ALL_FEATURES, how='left')[['release_date_x'] + ALL_FEATURES].copy().rename(columns={"release_date_x": "release_date"}).values
            # repeat the same process for the visualization dataframe
            # Determine indices where all TEST_FEATURES match
            small_mask_b_c = vis_df[TEST_FEATURES].apply(tuple, axis=1).isin(table_df[TEST_FEATURES].apply(tuple, axis=1))
            # Get indices where all columns in vis_df are not null
            small_true_indices = vis_df.notnull().all(axis=1)
            # Extract indices from vis_df where conditions are met
            small_indices_in_df2 = vis_df.index[small_mask_b_c & small_true_indices].tolist()
            for i in range(0, len(small_indices_in_df2)):
                vis_df.loc[small_indices_in_df2[i], CHANGE_FEATURES] = table_df[CHANGE_FEATURES].iloc[i].tolist()

        # create new reduction
        new_X_reduced = tsne.fit_transform(data[NUM_FEATURES])
        data['x_coor'], data['y_coor'] = new_X_reduced[:,0], new_X_reduced[:,1]

    fig = px.scatter(data, x='x_coor', y='y_coor',
                 hover_data={'x_coor': False,
                             'y_coor': False,
                             'artist': True,
                             'track': True,
                             'release_date': True,
                             })

    # Update the trace for fig1 data to make its markers red and larger
    fig.data[0].marker.color = ['red' if i in query_idx else 'blue' for i in range(len(data))]
    fig.data[0].marker.size = [20 if i in query_idx else 5 for i in range(len(data))]

    fig.update_layout(clickmode='event+select')
    fig.update_traces(selectedpoints=vis_df.index)

    # zoom in correctly
    if relayout_data is not None:
        if 'xaxis.range[0]' in relayout_data and 'yaxis.range[0]' in relayout_data:
            fig.update_layout(xaxis=dict(range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]),
                            yaxis=dict(range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]))
        elif 'xaxis.range[0]' in relayout_data:
            fig.update_layout(xaxis=dict(range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]))
        elif 'yaxis.range[0]' in relayout_data:
            fig.update_layout(yaxis=dict(range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]))
        elif 'dragmode' in relayout_data:
            fig.update_layout(dragmode=relayout_data['dragmode'])

    return fig, json.dumps(data.to_dict("index")), vis_df.to_dict('records')

@callback(
    Output('container-button-timestamp', 'children'),
    Output('acc-vis', 'figure'),
    Output('x_pool-store', 'data'),
    Output('y_pool-store', 'data'),
    Output('query-store', 'data'), 
    Input('train-btn', 'n_clicks'),
    Input('x_pool-store', 'data'),
    Input('y_pool-store', 'data'),
)
def displayClick(btn1, X_pool, y_pool):
    if isinstance(X_pool, list):
        X_pool = np.array(X_pool)

    msg = 'Start labeling!'
    query_idx_array = []
    fig2 = px.scatter(np.transpose(np.array(performance_history)))

    # if the button is clicked, train the model
    if "train-btn" == ctx.triggered_id:
        # query by committee
        for comm_idx in range(n_comm):

            # feed the committee 10 new instances, and delete them from the pool afterwards
            n_queries = 10
            for idx in range(n_queries):
                query_idx, query_instance = comm_list[comm_idx].query(X_pool)
                query_idx_array.append(query_idx[0])
                comm_list[comm_idx].teach(X_pool[query_idx], np.array(y_pool[comm_idx])[query_idx])

                X_pool = np.delete(X_pool, query_idx, axis=0)
                for cidx in range(n_comm):
                    y_pool[cidx] = np.delete(y_pool[cidx], query_idx, axis=0)
                
                performance_history[comm_idx] = np.append(performance_history[comm_idx], get_mse(comm_list[comm_idx], X_pool, y_pool[comm_idx]))
        
        # show the MSE performance over time
        df_perf = pd.DataFrame(np.transpose(performance_history), columns=['danceability', 'energy'])
        fig2 = px.line(df_perf)

    return html.Div(msg), fig2, X_pool, y_pool, query_idx_array

if __name__ == '__main__':
    app.run(debug=True)