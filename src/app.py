from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx

import json
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
from copy import deepcopy

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.metrics import mean_squared_error

from modAL.models import ActiveLearner, Committee
from modAL.disagreement import vote_entropy_sampling
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

# create training dataset
train = pd.concat([X, target], axis=1)

# generate the pool
X_pool = deepcopy(train[NUM_FEATURES].to_numpy())
y_pool = deepcopy(train[TARGET].to_numpy())

# initializing Committee members
n_members = 3
learner_list = list()

for member_idx in range(n_members):
    # initial training data
    n_initial = 5
    train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
    X_train = X_pool[train_idx]
    y_train = y_pool[train_idx]

    # creating a reduced copy of the data with the known instances removed
    X_pool = np.delete(X_pool, train_idx, axis=0)
    y_pool = np.delete(y_pool, train_idx, axis=0)


    # initializing learner
    learner = ActiveLearner(
        estimator=MultiTaskLassoCV(cv=5),
        X_training=X_train, y_training=y_train
    )
    learner_list.append(learner)

# assembling the committee
committee = Committee(learner_list=learner_list, query_strategy=vote_entropy_sampling)

# USES AVERAGE ACCURACY, COULD USE A BETTER METRIC
def get_score(X, y):
    count = np.zeros((n_members, y.shape[1]))
    for learner_idx, learner in enumerate(committee):
        Y_pred = learner.predict(X)
        
        for i in range(0, len(Y_pred)):
            for j in range(0, y.shape[1]):
                if abs(Y_pred[i][j] - y[i][j]) < 0.1:
                    count[learner_idx][j] = count[learner_idx][j] + 1
                    
    return np.sum(count, axis=0) / (n_members * len(X))

def get_mse(X, Y_true):
    MSE_dance = 0
    MSE_energy = 0
    for learner_idx, learner in enumerate(committee): 
        Y_pred = learner.predict(X)
        MSE_dance = MSE_dance + mean_squared_error(Y_pred[:,0], Y_true[:,0])
        MSE_energy = MSE_energy + mean_squared_error(Y_pred[:,1], Y_true[:,1])

    return [MSE_dance / n_members, MSE_energy / n_members] 


unqueried_score = get_mse(train[NUM_FEATURES].to_numpy(), train[TARGET].to_numpy())
    
performance_history = [unqueried_score]

df_perf = pd.DataFrame(performance_history, columns=['danceability', 'energy'])
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
)
def update_data(table_data, vis_data, relayout_data):
    triggered_id = ctx.triggered_prop_ids
    print(triggered_id)

    if table_data is None:
        # If no data is selected, display the entire DataFrame
        table_df = data
    else:
        # otherwise, display the selected data
        table_df = pd.DataFrame(table_data)

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
        mask_a = data[mask_b_c][CHANGE_FEATURES].reset_index(drop=True) != table_df[CHANGE_FEATURES].reset_index(drop=True)

        if type(CHANGE_FEATURES) != str:
            mask_a = mask_a.any(axis=1)

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
            vis_df.loc[small_indices_in_df2[i], CHANGE_FEATURES] = table_df[CHANGE_FEATURES].iloc[i].tolist()

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

    if relayout_data is None:
        # xmin = min(data['x_coor'])*1.1
        # xmax = max(data['x_coor'])*1.1
        # ymin = min(data['y_coor'])*1.1
        # ymax = max(data['y_coor'])*1.1    
        pass
    else:
        if 'autosize' in list(relayout_data.keys()) or 'xaxis.autorange' in list(relayout_data.keys()) or 'yaxis.autorange' in list(relayout_data.keys()):
            # xmin = min(data['x_coor'])*1.1
            # xmax = max(data['x_coor'])*1.1
            # ymin = min(data['y_coor'])*1.1
            # ymax = max(data['y_coor'])*1.1 
            # fig.update_layout(xaxis=dict(range=[xmin, xmax]), yaxis=dict(range=[ymin, ymax]))
            pass
        elif 'xaxis.range[0]' in list(relayout_data.keys()):
            xmin = relayout_data['xaxis.range[0]']
            xmax = relayout_data['xaxis.range[1]']
            fig.update_layout(xaxis=dict(range=[xmin, xmax]))
        elif 'yaxis.range[0]' in list(relayout_data.keys()):
            ymin = relayout_data['yaxis.range[0]']
            ymax = relayout_data['yaxis.range[1]']
            fig.update_layout(yaxis=dict(range=[ymin, ymax]))
        elif 'xaxis.range[0]' in list(relayout_data.keys()) and 'yaxis.range[0]' in list(relayout_data.keys()):
            xmin = relayout_data['xaxis.range[0]']
            xmax = relayout_data['xaxis.range[1]']
            ymin = relayout_data['yaxis.range[0]']
            ymax = relayout_data['yaxis.range[1]']
            fig.update_layout(xaxis=dict(range=[xmin, xmax]), yaxis=dict(range=[ymin, ymax]))
        elif 'dragmode' in list(relayout_data.keys()):
            fig.update_layout(dragmode=relayout_data['dragmode'])
        else:
            pass

    return fig, json.dumps(data.to_dict("index")), vis_df.to_dict('records')

@callback(
    Output('container-button-timestamp', 'children'),
    Output('acc-vis', 'figure'),
    Output('x_pool-store', 'data'),
    Output('y_pool-store', 'data'),
    Input('train-btn', 'n_clicks'),
    Input('x_pool-store', 'data'),
    Input('y_pool-store', 'data'),
)
def displayClick(btn1, X_pool, y_pool):
    if X_pool is None:
        X_pool = train[NUM_FEATURES].to_numpy()
    if y_pool is None:
        y_pool = train[TARGET].to_numpy()

    if isinstance(X_pool, list):
        X_pool = np.array(X_pool)
    if isinstance(y_pool, list):
        y_pool = np.array(y_pool)

    msg = 'Start labeling!'
    fig2 = px.scatter(np.array(performance_history))

    if "train-btn" == ctx.triggered_id:
        # query by committee
        n_queries = 10
        uncertainty_scores = classifier_margin(committee, X_pool)
        query_idx = uncertainty_scores.argsort()[-n_queries:][::-1]
        print(query_idx)
        print(X_pool[query_idx])
        committee.teach(
            X=X_pool[query_idx].reshape(-1, 20),
            y=y_pool[query_idx].reshape(-1, 2)
        )
        performance_history.append(get_mse(train[NUM_FEATURES].to_numpy(), train[TARGET].to_numpy()))
        # remove queried instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

        df_perf = pd.DataFrame(performance_history, columns=['danceability', 'energy'])
        fig2 = px.line(df_perf)


        # df = pd.read_json(StringIO(data)).transpose()
        # if np.count_nonzero(df['preference']) < 3:
        #     msg = "Label at least 3 instances before training the model"
        # else:
        #     msg = "Training model..."
        #     X_train = df[(df[['preference']] != 0).all(axis=1)][NUM_FEATURES]
        #     y_train = df[(df[['preference']] != 0).all(axis=1)]['dance'].astype('int')
        #     X_test = df[(df[['preference']] == 0).all(axis=1)][NUM_FEATURES]
        #     y_test = df[(df[['preference']] == 0).all(axis=1)]['dance'].astype('int')

        #     rfc = RandomForestClassifier(n_estimators=100, random_state=0)
        #     rfc.fit(X_train, y_train)
        #     y_pred = rfc.predict(X_test)

        #     score = sum(y_pred == y_test) / len(y_test)
        #     msg = "Achieved accuracy: " + str(score)
    return html.Div(msg), fig2, X_pool, y_pool

if __name__ == '__main__':
    app.run(debug=True)