from dash import Dash, html, dcc, callback, Output, Input, ctx, State, no_update, ALL
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, normalize
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

import scipy as sp
from scipy.spatial.distance import pdist, squareform

from modAL.models import ActiveLearner, CommitteeRegressor
from modAL.disagreement import max_std_sampling

import pandas as pd
import numpy as np
from copy import deepcopy
import seaborn as sns
import random

from functions import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# load dataset, select only a subset 
df = pd.read_csv("src/data/tcc_ceds_music.csv")
df = df[df['release_date'] >= 1999]
df = df.rename(columns={"genre": "Genre", "topic": "Topic"})

# normalize the release date
scaler = MinMaxScaler()
norm_date = scaler.fit_transform(np.array(df['release_date']).reshape(-1, 1))
df['release_date'] = norm_date
df['preference'] = 0

# different feature arrays
TRAIN_FEATURES = ['release_date', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'blues', 'country', 'hip hop', 
                'jazz', 'pop', 'reggae', 'rock', 'feelings', 'music', 'night/time', 'obscene', 'romantic', 'sadness', 
                'violence', 'world/life']

PCP_FEATURES = ['release_date', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'Genre', 'Topic']

COMMON_FEATURES = ['release_date', 'loudness', 'acousticness', 'instrumentalness', 'valence']

# initialize pcp attributes
bins = [-0.1, 0.2, 0.4, 0.6, 0.8, 1]
labels = ['Very Bad', 'Bad', 'Neutral', 'Good', 'Very Good']
rl_labels = ['Very Old', 'Old', 'Neutral', 'New', 'Very New']

# remove duplicates
df = df.drop_duplicates(subset=COMMON_FEATURES).reset_index(drop=True)

# define X and Y frames
X = df[['release_date', 'Genre', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'Topic']]
Y = df[['danceability', 'energy', 'preference']]

# one hot encode the categorical features
X = one_hot_encode(X, 'Genre')
X = one_hot_encode(X, 'Topic')

# number of training iterations
iteration_count = 0
# number of iteration before reset
iteration_reset = 2

# initializing Committee members
n_comm = 3
comm_list = list()
query_idx = []
labeled_idx = []
# create a committee for each target
for comm_idx in range(n_comm):
    n_members = 3
    learner_list = list()
    # add members to each committee
    for member_idx in range(n_members):
        # initializing learner
        learner = ActiveLearner(
            estimator=Ridge(alpha=1.0),
        )
        learner_list.append(learner)

    # initializing committee    
    committee = CommitteeRegressor(
        learner_list=learner_list,
        query_strategy=max_std_sampling
    )    
    comm_list.append(committee)

# define land marks
lands = random.sample(range(0,X.to_numpy().shape[0],1),int(X.to_numpy().shape[0]**0.5))
lands = np.array(lands,dtype=int)

# normalize X
X_norm = normalize(X.to_numpy()[:,:5], axis=0, norm='l1')

# create dimensionality reduction
dm1 = pairwise_distances(X_norm, X_norm[lands], metric='cosine', n_jobs=-1)
jaccard_distances = pdist(X.to_numpy()[:,5:], metric='jaccard')
dm2 = squareform(jaccard_distances)[lands]

# normalize the dm's
dm1_norm = dm1 / np.max(dm1)
dm2_norm = dm2 / np.max(dm2)

# create dimensionality matrix for the numerical + categorical features
dm = 0.15 * dm1_norm + 0.15 * dm2_norm.T

# create dimensionality matrix for the prediction features
dm_pred = get_pred_dm(X, comm_list, n_comm, lands)

# get 2d coordinates from the dimensionality matrices
X_dm, y_dm = get_dm_coords(dm, dm_pred, lands)

# create new dataframe
data = X.copy()
data['x_coor'], data['y_coor'] = X_dm, y_dm
data['artist'], data['track'] = df[['artist_name']], df[['track_name']]
data['Genre'], data['Topic'] = df['Genre'], df['Topic']
data['manual_dance'], data['manual_ener'], data['manual_pref'] = -1, -1, -1

# create training dataset
train = pd.concat([X, Y], axis=1)

# generate the pool
X_pool = deepcopy(train[TRAIN_FEATURES].to_numpy())
y_pool1 = deepcopy(train['danceability'].to_numpy())
y_pool2 = deepcopy(train['energy'].to_numpy())
y_pool3 = deepcopy(train['preference'].to_numpy())
y_pool = [y_pool1, y_pool2, y_pool3]

# create a plot for the model performance over time
perf_hist = [[-1, 2] for i in range(n_comm)]
performance_history = [[-1] + [None] * len(data) + [2] for _ in perf_hist]
fig2 = create_density_plot(performance_history)

# turn string into processable string
def format_string(string):
    # split the string into separate lines
    lines = string.split("<br>")

    # initialize variables
    artist = None
    track = None

    # iterate over each line and extract information
    for line in lines:
        if line.startswith("Artist:"):
            artist = line.split("Artist: ")[1].strip()
        elif line.startswith("Track:"):
            track = line.split("Track: ")[1].strip()

    danceability = df[(df['artist_name'] == artist) & (df['track_name'] == track)]['danceability'].values[0]
    energy = df[(df['artist_name'] == artist) & (df['track_name'] == track)]['energy'].values[0]

    # construct the formatted string
    formatted_string = f"{track.title()} by {artist.title()}, [{round(danceability, 2)}, {round(energy, 2)}]"
    
    return formatted_string

# get model prediction for a certain artist + track
def get_pred(artist, track):
    # retrieve data sample
    x = data[(data['artist'] == artist) & (data['track'] == track)][TRAIN_FEATURES]
    
    # check if model is fitted
    model_fitted = True
    for idx, committee in enumerate(comm_list):
        if not are_all_learners_fitted(committee):
            model_fitted = False

    # get prediction    
    preds = []
    if model_fitted:
        for comm_idx in range(n_comm):
            pred = comm_list[comm_idx].predict(x.to_numpy())
            preds.append(pred[0])
        return preds
        
    else:
        return [0, 0, 0]

# create scatterplot
fig = px.scatter(data, x='x_coor', y='y_coor', 
                                hover_data={'x_coor':False, 
                                            'y_coor':False,
                                            'artist':True,
                                            'track':True,
                                            'release_date':True,
                                            })
fig = add_custom_legend(fig)
fig.update_layout(clickmode='event+select', margin=dict(l=20, r=20, t=20, b=20),)

# create a mock-up pcp
MT = np.zeros(len(data.columns))
df_pcp = pd.DataFrame([MT], columns=data.columns)
pcp = create_pcp(df_pcp)

# dashboard layout
app = Dash(__name__)
app.layout = html.Div([
    # container for the title and the left-side block
    html.Div(style={'width': '100%', 'float': 'left', 'margin-left': '2%'}, children=[
        # text above the left-side block
        html.H1("Interactive Visualization for Multi-Task AL", style={'margin-left': '2%', 'margin-right': '2%'}),    
        html.Div(className='left-side', style={
                'width': '90%', 
                'height':'575px',
                'margin-left':'1%', 
                'margin-right': '1%', 
                'margin-bottom': '1%',
                'display': 'inline-block', 
                'border': '1px solid black', 
                'boxShadow': 'rgba(0, 0, 0, 0.12) 0px 1px 3px, rgba(0, 0, 0, 0.24) 0px 1px 2px',
                'borderRadius': '10px',  # add rounded edges
                'backgroundColor': '#f9f9f9',
                'padding': '10px'
            }, children=[
            html.H2("Data Visualization:"),
            dbc.Col(dcc.Graph(id='main-vis', figure=fig)),
            html.Div([
                html.Button('Download CSV', id='download-btn'),
                dcc.Download(id='download-dataframe-csv'),
            ]),
        ]),
    ]),

    html.Div(className='right-side', style={'width': '100%', 'float': 'left'}, children=[
        # container for the top-right and bottom-right sections
        html.Div(style={'width': '100%', 'float': 'left', 'margin-left': '1%', 'margin-right': '1%'}, children=[
            # PCP plot
            html.Div(className='top-right', children=[
                html.H2("Data Attributes:"),
                dcc.Graph(id='pcp-vis', figure=pcp),
            ], style={
                'height':'200px',
                'margin-left':'1%', 
                'margin-right': '1%',
                'border': '1px solid black', 
                'boxShadow': 'rgba(0, 0, 0, 0.12) 0px 1px 3px, rgba(0, 0, 0, 0.24) 0px 1px 2px',
                'borderRadius': '10px',  # add rounded edges
                'backgroundColor': '#f1f1f1',
                'padding': '10px',
                'marginBottom': '1%', 
            }),

            # slider, checkboxes, and button
            html.Div(className='bottom-right', style={
                'height':'425px',
                'margin-left':'1%', 
                'margin-right': '1%',
                'border': '1px solid black', 
                'boxShadow': 'rgba(0, 0, 0, 0.12) 0px 1px 3px, rgba(0, 0, 0, 0.24) 0px 1px 2px',
                'borderRadius': '10px',  # add rounded edges
                'backgroundColor': '#f1f1f1',
                'padding': '10px'
            }, children=[
                html.Div(className='right-left-side', style={
                    'height':'400px',
                    'width': '48%',
                    'float': 'left',
                    'margin-bottom': '1%',
                    'padding': '5px',
                }, children=[
                    html.H2("Controls:"),
                    html.Div(children=[
                        html.Label('Danceability:', style={'font-weight': 'bold', 'font-size':'13px'}),
                        dcc.Slider(min=0, max=1, value=0.5, 
                                marks={0: 'Very Bad', 0.25: 'Bad', 0.5: 'Neutral', 0.75: 'Good', 1: 'Very Good'}, 
                                id='dance-slider')
                    ]),
                    html.Div(children=[
                        html.Label('Energy:', style={'font-weight': 'bold', 'font-size':'13px'}),
                        dcc.Slider(min=0, max=1, value=0.5, 
                                marks={0: 'Very Bad', 0.25: 'Bad', 0.5: 'Neutral', 0.75: 'Good', 1: 'Very Good'},
                                id='energy-slider')
                    ]),
                    html.Div(children=[
                        html.Label('Preference:', style={'font-weight': 'bold', 'font-size':'13px'}),
                        dcc.Slider(min=0, max=1, value=0.5, 
                                marks={0: 'Very Bad', 0.25: 'Bad', 0.5: 'Neutral', 0.75: 'Good', 1: 'Very Good'}, 
                                id='pref-slider')
                    ]),
                    html.Div(id='checklist-output', style={"minHeight": "150px", "maxHeight": "150px", "overflow-y": "scroll"}),
                    html.Button('Train the model!', id='train-btn', n_clicks=0, style={'margin':'1px'}), 
                    html.Button('Remove selected items', id='remove-btn', n_clicks=0, style={'margin':'1px'}),
                ]),
                # performance plot
                html.Div(className='right-right-side', style={
                    'height':'400px',
                    'width': '48%',
                    'float': 'right',
                    'margin-bottom': '1%',
                    'padding': '5px',
                }, children=[
                    html.H2("Labeling Distribution:"),
                    html.Div(children=[
                        dcc.Graph(id='acc-vis', figure=fig2), 
                    ]), 
                ])
            ]),
        ]),
    ]),

    # stored variables
    dcc.Store(id='data-store'),
    dcc.Store(id='x_pool-store', data=X_pool),
    dcc.Store(id='y_pool-store', data=y_pool),
    dcc.Store(id='query-store', data=query_idx),
    dcc.Store(id='labeled-store', data=labeled_idx),    
    dcc.Store(id='pcp-store'),
    dcc.Store(id='reset-bool', data=False),
    dcc.Store(id='iteration-count', data=iteration_count),
], style={"display": "flex", "flexDirection": "row", "justifyContent": "space-between", "alignItems": "flex-start", "padding": "20px"})


# callback to update the scatters in the main visualization
@app.callback(
    Output('main-vis', 'figure'),
    Output('reset-bool', 'data', allow_duplicate=True),
    Input('query-store', 'data'),
    Input('labeled-store', 'data'),
    State('checklist-output', 'children'),
    State('main-vis', 'relayoutData'),
    Input('reset-bool', 'data'),
    State('iteration-count', 'data'),
    prevent_initial_call='initial_duplicate'
)
def update_plot(query_idx, labeled_idx, current_children, relayout_data, reset_bool, iteration_count):
    blue_color_scale = [
        [0, '#add8e6'],  # light blue
        [1, '#00008b']   # dark blue
    ]

    # check if the AL models are fitted
    model_fitted = True
    for idx, committee in enumerate(comm_list):
        if not are_all_learners_fitted(committee):
            model_fitted = False

    preds = [1 for i in range(len(data))]

    # if model is fitted, get preference prediction
    if model_fitted:
        preds = comm_list[2].predict(X_pool)

    # if the training button is pressed
    if reset_bool:
        # and the number of iterations is divible by a set amount
        if iteration_count % iteration_reset == 0:
            # recalculate dimensionality coordinates
            dm_pred = get_pred_dm(X, comm_list, n_comm, lands)
            X_dm, y_dm = get_dm_coords(dm, dm_pred, lands)
            data['x_coor'], data['y_coor'] = X_dm, y_dm
        reset_bool = False

    data_copy = data.copy()
    data_copy['preds'] = preds

    # retrieve the data
    preds_data = data_copy[~data_copy.index.isin(labeled_idx)]['preds']
    x_data = data_copy[~data_copy.index.isin(labeled_idx)]['x_coor']
    y_data = data_copy[~data_copy.index.isin(labeled_idx)]['y_coor']
    artist_data = data_copy[~data_copy.index.isin(labeled_idx)]['artist']
    track_data = data_copy[~data_copy.index.isin(labeled_idx)]['track']
    release_date_data = data_copy[~data_copy.index.isin(labeled_idx)]['release_date']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # create the default scatter plot
    scatter_trace = go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        hoverinfo='text',
        text=[f'Artist: {a}<br>Track: {t}<br>Release Date: {d}' for a, t, d in zip(artist_data, track_data, release_date_data)],
        marker=dict(
            color=preds_data,
            colorscale=blue_color_scale,  # use the custom blue color scale
            size=5
        ),
        showlegend=False,
        )
    
    fig.add_trace(scatter_trace, secondary_y=False)

    if labeled_idx[:-0 or None]:
        # retrieve the already labeled data
        label_idx = labeled_idx[:-0 or None]
        x_data = data.iloc[label_idx]['x_coor']
        y_data = data.iloc[label_idx]['y_coor']
        artist_data = data.iloc[label_idx]['artist']
        track_data = data.iloc[label_idx]['track']
        release_date_data = data.iloc[label_idx]['release_date']

        # create the scatter plot for the already labeled data
        scatter_trace = go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            hoverinfo='text',
            text=[f'Artist: {a}<br>Track: {t}<br>Release Date: {d}' for a, t, d in zip(artist_data, track_data, release_date_data)],
            marker=dict(
                color='red',
            ),
            showlegend=False
        )
        fig.add_trace(scatter_trace, secondary_y=False)

    # create the scatter plot for the AL guidance scatters
    if query_idx:
        glyph_fig = plot_bar_chart_glyphs_from_dataframe(data.iloc[query_idx], n_comm, comm_list, TRAIN_FEATURES)
        for trace in glyph_fig.data:
            fig.add_trace(trace, secondary_y=True)
    fig.update_layout(clickmode='event+select', margin=dict(l=20, r=20, t=20, b=20))

    # zoom in correctly
    if relayout_data is not None:
        if 'xaxis.range[0]' in relayout_data and 'yaxis.range[0]' in relayout_data:
            fig.update_layout(xaxis=dict(range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]),
                            yaxis=dict(range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]))
        elif 'xaxis.range[0]' in relayout_data:
            fig.update_layout(xaxis=dict(range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]))
        elif 'yaxis.range[0]' in relayout_data:
            fig.update_layout(yaxis=dict(range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]))

    fig = add_custom_legend(fig)

    # find out which samples are selected by the user
    indices = []
    indices_2 = []
    if current_children:
        selected_labels = [item['props']['options'][0]['value'] for item in current_children]
        for i in selected_labels:
            if i in fig.data[0]['text']:
                index = fig.data[0]['text'].index(i)
                indices.append(index)
            if fig.data[1]['text']:
                if i in fig.data[1]['text']:
                    index = fig.data[1]['text'].index(i)
                    indices_2.append(index)

    # select the labels in the plot
    if indices:
        for i, trace in enumerate(fig.data):
            if i == 0:
                trace.selectedpoints = indices
    if indices_2:
        for i, trace in enumerate(fig.data):
            if i == 1:
                trace.selectedpoints = indices_2
    elif indices and not indices_2:
        for i, trace in enumerate(fig.data):
            if i == 1:
                trace.selectedpoints = []

    # combine layers
    fig.update_layout(
        yaxis2=dict(overlaying='y', layer='below traces')
    )

    # this ensures the ranges match
    fig.update_yaxes(matches='y')
    fig.update_xaxes(matches='x')

    # remove background color
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # remove axes
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_yaxes(visible=False, showticklabels=False)

    return fig, reset_bool

# callback for the AL lasso selection
@callback(
    Output('checklist-output', 'children', allow_duplicate=True),
    Output('pcp-vis', 'figure', allow_duplicate=True),
    Input('main-vis', 'selectedData'),
    State('checklist-output', 'children'),
    prevent_initial_call='initial_duplicate'
)
def lasso_select(selected_data, current_children):
    if selected_data is None:
        return no_update
    
    points = selected_data.get('points', [])
    if not points:
        return no_update
    
    lasso = selected_data.get('lassoPoints', [])
    if not lasso:
        return no_update
    
    # check if the AL models are fitted
    model_fitted = True
    for idx, committee in enumerate(comm_list):
        if not are_all_learners_fitted(committee):
            model_fitted = False

    # create a dataframe for the points inside the lasso selection
    df_al = pd.DataFrame(columns=data.columns)
    filtered_points = [point for point in selected_data['points'] if point['curveNumber'] == 0]
    for i in filtered_points:
        artist, track, rl = string_to_var(i['text'])
        filtered_data = data[(data['artist'] == artist) & (data['track'] == track)].dropna(axis=1, how='all')
        df_al_filtered = df_al.dropna(axis=1, how='all')
        df_al = pd.concat([df_al_filtered, filtered_data], ignore_index=True)

    # initialize variables
    x_df = df_al[TRAIN_FEATURES].to_numpy()
    std_list = np.zeros(len(x_df))

    # if the model is fitted, find the 5 most usefull points according to the AL model
    if model_fitted:
        for comm_idx in range(n_comm):
            _, stds = comm_list[comm_idx].predict(x_df, return_std=True)
            std_list = [x + y for x, y in zip(std_list, stds)]

        # convert std_list to a numpy array for efficient indexing
        std_array = np.array(std_list)

        # number of elements to find
        n_largest = 5

        # handle case where std_list has fewer than n_largest elements
        if len(std_array) <= n_largest:
            sorted_largest_indices = np.argsort(-std_array)
        else:
            # get the indices of the n_largest elements
            largest_indices = np.argpartition(std_array, -n_largest)[-n_largest:]
            # sort these indices by their corresponding values in descending order
            sorted_largest_indices = largest_indices[np.argsort(-std_array[largest_indices])]

        # convert numpy array to list if needed
        query_idx = sorted_largest_indices.tolist()

    # if the AL model is not fitted, randomly select 5 samples from the lasso selection
    if not model_fitted:
        query_idx = np.random.randint(0, len(df_al), size=5)
        query_idx = list(set(query_idx))

    # merge selected dataframe with the whole dataframe
    merged_df = data.merge(df_al.iloc[query_idx], on=COMMON_FEATURES, how='outer', indicator=True)

    # filter rows that are in both DataFrames
    matching_rows = merged_df[merged_df['_merge'] == 'both']

    # drop the indicator column if not needed
    matching_rows = matching_rows.drop(columns=['_merge'])
    samples = matching_rows[['artist_y', 'track_y', 'release_date']]

    if current_children is None:
        current_children = []
    df_pcp = pd.DataFrame(columns=data.columns)

    # for each sample, add to checklist and pcp
    for index, row in samples.iterrows():
        artist = row['artist_y']
        track = row['track_y']
        rl = row['release_date']
        
        sample = var_to_string(artist, track, rl)
        
        new_item = dcc.Checklist(
            id={'type': 'data-checklist', 'label': sample},
            options=[{'label': format_string(sample), 'value': sample}],
            value=[sample],
            inline=True
        )
        current_children = current_children + [new_item]

        # add item to pcp
        filtered_data = data[(data['artist'] == artist) & (data['track'] == track)].dropna(axis=1, how='all')
        df_pcp_filtered = df_pcp.dropna(axis=1, how='all')
        df_pcp = pd.concat([df_pcp_filtered, filtered_data], ignore_index=True)

    # create pcp
    pcp = create_pcp(df_pcp)

    return current_children, pcp

# callback for the removal of selected items from the labeling list
@callback(
    Output('checklist-output', 'children', allow_duplicate=True),
    Output('pcp-vis', 'figure', allow_duplicate=True),
    Output('pcp-store', 'data', allow_duplicate=True),
    Input('remove-btn', 'n_clicks'),
    State('checklist-output', 'children'),
    State('pcp-store', 'data'), 
    prevent_initial_call='initial_duplicate'
)
def remove_items(btn2, current_children, pcp_df):
    if "remove-btn" == ctx.triggered_id:
        pcp_df = pd.DataFrame(pcp_df)
        value = [item['props']['value'] for item in current_children]
        
        # remove all items that are checked from the dataframe
        for i in value:
            if i:
                artist, track, rl = string_to_var(i[0])
                pcp_df = pcp_df[(pcp_df.artist != artist) & (pcp_df.track != track)]

        pcp = create_pcp(pcp_df)

        # keep only the children that are not checked
        current_children = [i for i in current_children if not i['props']['value']]
        return current_children, pcp, pcp_df.to_dict('records')
    
    return no_update

# callback for the button, starts new model iteration
@callback(
    Output('acc-vis', 'figure'),
    Output('x_pool-store', 'data'),
    Output('y_pool-store', 'data'),
    Output('query-store', 'data'), 
    Output('reset-bool', 'data', allow_duplicate=True),
    Output('iteration-count', 'data'),
    Input('train-btn', 'n_clicks'),
    State('x_pool-store', 'data'),
    State('y_pool-store', 'data'),
    State('pcp-store', 'data'),
    State('dance-slider', 'value'),
    State('energy-slider', 'value'),
    State('pref-slider', 'value'),
    State('checklist-output', 'children'),
    State('query-store', 'data'),
    State('iteration-count', 'data'),
    prevent_initial_call='initial_duplicate'
)
def train_model(btn1, X_pool, y_pool, df, danceability, energy, preference, current_children, query_idx, iteration_count):
    if isinstance(X_pool, list):
        X_pool = np.array(X_pool)

    # initialize variables
    reset_bool = False
    fig2 = create_density_plot(performance_history)

    y_pool_test = [danceability, energy, preference]

    if not df:
        return fig2, X_pool, y_pool, query_idx, reset_bool, iteration_count

    # if the button is clicked, train the model
    if "train-btn" == ctx.triggered_id:
        if current_children:
            value = [item['props']['value'] for item in current_children]
            if not all(not sublist for sublist in value):
                reset_bool = True

        if not reset_bool:
            return fig2, X_pool, y_pool, query_idx, reset_bool, iteration_count

        artist = pd.DataFrame(df)['artist'].tolist()
        track = pd.DataFrame(df)['track'].tolist()
        indices = data[(data['artist'].isin(artist)) & (data['track'].isin(track))].index.tolist()
        # query by committee
        std_list = np.zeros(len(X_pool))

        # train the model on the new instances
        for comm_idx in range(n_comm):
            comm_list[comm_idx].teach(X_pool[indices], np.array([y_pool_test[comm_idx]] * X_pool[indices].shape[0]))

        # get new predictions and performance
        for comm_idx in range(n_comm):
            _, stds = comm_list[comm_idx].predict(X_pool, return_std=True)
            std_list = [x + y for x, y in zip(std_list, stds)]
            for i in indices:
                performance_history[comm_idx][i+1] = y_pool_test[comm_idx]
            
        # convert std_list to a numpy array for efficient indexing
        std_array = np.array(std_list)

        # number of elements to find
        n_largest = 5

        # handle case where std_list has fewer than n_largest elements
        if len(std_array) <= n_largest:
            sorted_largest_indices = np.argsort(-std_array)
        else:
            # get the indices of the n_largest elements
            largest_indices = np.argpartition(std_array, -n_largest)[-n_largest:]
            # sort these indices by their corresponding values in descending order
            sorted_largest_indices = largest_indices[np.argsort(-std_array[largest_indices])]

        # convert numpy array to list if needed
        query_idx = sorted_largest_indices.tolist()
        
        # create performance plot
        fig2 = create_density_plot(performance_history)

        iteration_count = iteration_count + 1

    return fig2, X_pool, y_pool, query_idx, reset_bool, iteration_count

# callback for the labeling interface
@app.callback(
    Output('checklist-output', 'children', allow_duplicate=True),
    Output('pcp-vis', 'figure', allow_duplicate=True),
    Output('pcp-store', 'data', allow_duplicate=True),
    Output('reset-bool', 'data', allow_duplicate=True), 
    Output('labeled-store', 'data', allow_duplicate=True),
    Output('main-vis', 'clickData'),
    Input('main-vis', 'clickData'),
    Input('reset-bool', 'data'),
    Input('labeled-store', 'data'),
    State('pcp-store', 'data'),    
    State('query-store', 'data'),
    Input('checklist-output', 'children'),
    Input({'type': 'data-checklist', 'label': ALL}, 'value'),
    prevent_initial_call='initial_duplicate'
)
def handle_labeling(click_data, reset_bool, labeled_idx, pcp_df, query_idx, current_children, check_values):
    MT = np.zeros(len(data.columns))
    df_pcp = pd.DataFrame([MT], columns=data.columns)
    pcp = create_pcp(df_pcp)
    
    # if the button is pressed, remove check items, keep unchecked items
    new_children = current_children
    if reset_bool:
        pcp_df = pd.DataFrame(pcp_df)

        # if checkbox is checked, the item gets a 'value'
        value = [item['props']['value'] for item in current_children]
        
        # remove all items that are checked from the dataframe
        for i in value:
            if i:
                artist, track, rl = string_to_var(i[0])
                pcp_df = pcp_df[(pcp_df.artist != artist) & (pcp_df.track != track)]

                labeled_idx = labeled_idx + data[(data['artist'] == artist) & (data['track'] == track)].index.tolist()

        # create a new figure for the unchecked data
        pcp = create_pcp(pcp_df)

        # keep only the children that are not checked
        new_children = [i for i in current_children if not i['props']['value']]
    
    # if no data is clicked, return a mock-up pcp
    if not click_data:
        if not new_children:            
            if not isinstance(pcp_df, list) and pcp_df is not None:
                pcp_df = pcp_df.to_dict('records')
            return new_children, pcp, pcp_df, reset_bool, labeled_idx, None
        else:
            # if item is unchecked, remove from pcp plot, but keep in checklist
            pcp_df = pd.DataFrame(columns=data.columns)
            for i in check_values:
                if i:
                    artist, track, rl = string_to_var(i[0])
                    row = data[(data['artist'] == artist) & (data['track'] == track)]
                    pcp_df = pd.concat([pcp_df, row])
                    pcp = create_pcp(pcp_df)
            return new_children, pcp, pcp_df.to_dict('records'), reset_bool, labeled_idx, None

    # find and initialize the samples
    artist, track, release_date, sample = None, None, None, None
    if click_data['points'][0]['curveNumber'] == 0:
        sample = click_data['points'][0]['text']
        artist, track, release_date = string_to_var(sample)

    elif click_data['points'][0]['curveNumber'] == 1:
        id = click_data['points'][0]['pointIndex']
        queried = labeled_idx[id]
        artist = data.iloc[queried]['artist']
        track = data.iloc[queried]['track']
        release_date = data.iloc[queried]['release_date']
        sample = var_to_string(artist, track, release_date)

    elif click_data['points'][0]['curveNumber'] == 2:
        id = click_data['points'][0]['pointNumber']
        queried = query_idx[id]
        artist = data.iloc[queried]['artist']
        track = data.iloc[queried]['track']
        release_date = data.iloc[queried]['release_date']
        sample = var_to_string(artist, track, release_date)

    elif click_data['points'][0]['curveNumber'] > 2:
        id = (click_data['points'][0]['curveNumber'] - 5) // 5
        queried = query_idx[id]
        artist = data.iloc[queried]['artist']
        track = data.iloc[queried]['track']
        release_date = data.iloc[queried]['release_date']
        sample = var_to_string(artist, track, release_date)

    # create a dataframe for all the clicked samples
    vis_df = data[(data['artist'].isin([artist])) & (data['track'].isin([track]))]

    if current_children is None:
        current_children = []

    if pcp_df is None:
        pcp_df = pd.DataFrame(columns=vis_df.columns)

    # extract selected labels from current checklist items
    selected_labels = [item['props']['options'][0]['label'] for item in current_children]

    # check if the clicked label is already selected
    if format_string(sample) in selected_labels:
        # if clicked label is selected, remove it from the checklist items
        new_children = [item for item in current_children if item['props']['options'][0]['label'] != format_string(sample)]
    else:
        # if clicked label is not selected, add it to the checklist items
        new_item = dcc.Checklist(
            id={'type': 'data-checklist', 'label': sample},
            options=[{'label': format_string(sample), 'value': sample}],
            value=[sample],
            inline=True
        )
        new_children = current_children + [new_item]

    pcp_df = pd.DataFrame(pcp_df)

    # check conditions and create the result DataFrame
    if pcp_df.empty and not vis_df.empty:
        result = vis_df.copy()  # if pcp is empty but vis has rows, result is vis
    elif pcp_df.equals(vis_df):
        result = pd.DataFrame()  # if pcp and vis are the same, result is an empty DataFrame
    else:
        # find rows that are different between pcp and vis
        diff_rows = pd.concat([pcp_df, vis_df]).drop_duplicates(keep=False)
        result = diff_rows

    pcp_df = result
    pcp_df = pcp_df.drop_duplicates(subset=COMMON_FEATURES, ignore_index=True, keep='first').copy()

    dataframe = pd.DataFrame(columns = pcp_df.columns)
    for item in new_children:
        artist, track, rl = None, None, None
        if isinstance(item, dict):
            artist, track, rl = string_to_var(item.get('props', {})['options'][0].get('value'))
        elif isinstance(item, dcc.Checklist):
            artist, track, rl = string_to_var(item.value[0])
        for index, row in pcp_df.iterrows():
            if row['track'] == track and row['artist'] == artist:
                dataframe = pd.concat([dataframe, pd.DataFrame([row])], ignore_index=True)

    # create a new pcp with the resulting df
    new_pcp = create_pcp(dataframe)
    return new_children, new_pcp, dataframe.to_dict('records'), reset_bool, labeled_idx, None

# callback to download a csv with the original data and the new model predictions
@callback(
    Output("download-dataframe-csv", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_csv(n_clicks):
    # check if the AL models are fitted
    model_fitted = True
    for idx, committee in enumerate(comm_list):
        if not are_all_learners_fitted(committee):
            model_fitted = False

    preds = []

    if not model_fitted:
        return dcc.send_data_frame(data.to_csv, "mydf.csv")
    
    # make model predictions
    if model_fitted:
        for comm_idx in range(n_comm):
            pred = comm_list[comm_idx].predict(X.to_numpy())
            preds.append(pred)

    # add predictions to list
    for comm_idx in range(0, n_comm):
        for j in range(0, len(X)):
            if performance_history[comm_idx][j+1] != -1:
                performance_history[comm_idx][j+1] = preds[comm_idx][j]

    result = [sublist[1:-1] for sublist in performance_history]

    data[['manual_dance', 'manual_ener', 'manual_pref']] = pd.DataFrame(result).T

    return dcc.send_data_frame(data.to_csv, "mydf.csv")

if __name__ == '__main__':
    app.run(debug=True)