from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx, State, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

import json
import random
import math
import heapq
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from copy import deepcopy
from datetime import datetime


from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

from modAL.models import ActiveLearner, CommitteeRegressor
from modAL.disagreement import max_std_sampling

df = pd.read_csv("src/data/tcc_ceds_music.csv")
df = df[df['release_date'] >= 1999]

scaler = MinMaxScaler()
norm_date = scaler.fit_transform(np.array(df['release_date']).reshape(-1, 1))
df['release_date'] = norm_date
df['preference'] = 0

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

COMMON_FEATURES = ['release_date', 'loudness', 'acousticness', 'instrumentalness', 'valence']

CHANGE_FEATURES = ['release_date', 'preference']

df = df.drop_duplicates(subset=COMMON_FEATURES).reset_index(drop=True)

X = df[['release_date', 'genre', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'topic']]
Y = df[['danceability', 'energy', 'preference']]

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
data['x_coor'], data['y_coor'] = X_reduced[:,0]*20, X_reduced[:,1]*20
data['artist'], data['track'] = df[['artist_name']], df[['track_name']]
data['preference'] = 0

# create training dataset
train = pd.concat([X, Y], axis=1)

# generate the pool
X_pool = deepcopy(train[['release_date'] + TEST_FEATURES].to_numpy())
y_pool1 = deepcopy(train['danceability'].to_numpy())
y_pool2 = deepcopy(train['energy'].to_numpy())
y_pool3 = deepcopy(train['preference'].to_numpy())
y_pool = [y_pool1, y_pool2, y_pool3]

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

# create a plot for the MSE over time
performance_history = [[0] for i in range(n_comm)]
for comm_idx in range(n_comm):
    performance_history[comm_idx] = [None]

df_perf = pd.DataFrame(np.transpose(performance_history), columns=['danceability', 'energy', 'preference'])
fig2 = px.line(df_perf)

def var_to_string(artist, track, release_date):
    # Construct the string with the provided information
    string = f"Artist: {artist}<br>Track: {track}<br>Release Date: {release_date}"    
    return string

def string_to_var(string):
    # Split the string into separate lines
    lines = string.split("<br>")

    # Initialize variables
    artist = None
    track = None
    release_date = None

    # Iterate over each line and extract information
    for line in lines:
        if line.startswith("Artist:"):
            artist = line.split("Artist: ")[1].strip()
        elif line.startswith("Track:"):
            track = line.split("Track: ")[1].strip()
        elif line.startswith("Release Date:"):
            release_date = float(line.split("Release Date: ")[1])

    return artist, track, release_date

def format_string(string):
    # Split the string into separate lines
    lines = string.split("<br>")

    # Initialize variables
    artist = None
    track = None

    # Iterate over each line and extract information
    for line in lines:
        if line.startswith("Artist:"):
            artist = line.split("Artist: ")[1].strip()
        elif line.startswith("Track:"):
            track = line.split("Track: ")[1].strip()

    # Construct the formatted string
    formatted_string = f"'{track}' by '{artist}'"
    
    return formatted_string

def plot_star_glyphs_from_dataframe(data):
    """
    Plot non-overlapping star glyphs with filled enclosed areas and center points from a DataFrame.

    Args:
    - data (pandas.DataFrame): DataFrame containing the following columns:
                                'x_coord': x coordinates for the center points of the star glyphs.
                                'y_coord': y coordinates for the center points of the star glyphs.
                                'glyph_values': List of lists containing values for each glyph.
                                                Each inner list represents the values for a single glyph.
                                'artist': Artist name for each glyph.
                                'track_name': Track name for each glyph.

    Returns:
    - fig (plotly.graph_objs.Figure): Plotly figure containing the star glyphs.
    """

    fig = px.scatter(data, x='x_coor', y='y_coor', 
                                hover_data={'x_coor':False, 
                                            'y_coor':False,
                                            'artist':True,
                                            'track':True,
                                            'release_date':True,
                                            })

    # Add traces for each star glyph
    for i, row in data.iterrows():
        x = row['x_coor']
        y = row['y_coor']
        values = [0 for i in range(n_comm)]

        for comm_idx in range(n_comm):
            values[comm_idx] = comm_list[comm_idx].predict([row[NUM_FEATURES].tolist()])

        num_variables = len(values)
        angles = np.linspace(0, 2*np.pi, num_variables, endpoint=False)

        # Calculate coordinates for the arms
        x_arms = x + values * np.cos(angles)
        y_arms = y + values * np.sin(angles)

        # Close the shape
        x_arms = np.append(x_arms, x_arms[0])
        y_arms = np.append(y_arms, y_arms[0])

        # Add filled area trace
        fig.add_trace(
            go.Scatter(
                x=x_arms,
                y=y_arms,
                fill='toself',
                fillcolor='rgba(0, 255, 0, 1)',  # Red fill color with transparency
                line=dict(color='rgba(0, 255, 0, 0)'),  # Hide line
                showlegend=False,
            )
        )

        # Calculate coordinates for the circle
        circle_angles = np.linspace(0, 2*np.pi, 100)
        circle_x = x + np.cos(circle_angles)
        circle_y = y + np.sin(circle_angles)

        # Add trace for the circle
        fig.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='lines',
                line=dict(color='rgba(0, 255, 0, 1)'),  # Black circle line with full opacity
                showlegend=False,
            )
        )
    return fig

def is_model_fitted(model):
    try:
        # Check if the model is fitted by checking one of the fitted attributes
        check_is_fitted(model)
        return True
    except NotFittedError:
        return False
    
def are_all_learners_fitted(committee):
    for learner in committee.learner_list:
        if not is_model_fitted(learner.estimator):
            return False
    return True


# create scatterplot
fig = px.scatter(data, x='x_coor', y='y_coor', 
                                hover_data={'x_coor':False, 
                                            'y_coor':False,
                                            'artist':True,
                                            'track':True,
                                            'release_date':True,
                                            },
                                width=600, height=600)

fig.update_layout(clickmode='event+select', margin=dict(l=20, r=20, t=20, b=20),)

MT = np.zeros(len(data.columns))
df_pcp = pd.DataFrame([MT], columns=data.columns)
pcp = go.Figure(data=
    go.Parcoords(dimensions=[dict(range=[0, 1], label=feature, values=df_pcp[feature]) for feature in TEST_FEATURES])
)
pcp.update_layout(
    height=300,  # specify the height
    width=800    # specify the width
)
pcp.update_traces(
    dimensions=[
        {**d, **{"tickvals": np.linspace(0, 1, 5)}}
        for d in pcp.to_dict()["data"][0]["dimensions"]
    ]
)

fig2.update_layout(
    height=300,  # specify the height
    width=350    # specify the width
)

# Layout
app = Dash(__name__)
app.layout = html.Div([
    html.Div(className='left-side', style={'width': '50%', 'display': 'inline-block'}, children=[
        html.H2("Data visualization"),
        dbc.Col(dcc.Graph(id='main-vis', figure=fig)),
    ]),

    html.Div(className='right-side', style={'width': '50%', 'display': 'inline-block'}, children=[
        html.Div(className='top-right', children=[
            html.H3("Data Attributes"),
            dcc.Graph(id='pcp-vis', figure=pcp),
        ]),

        html.Div(className='bottom-right', children=[
            html.Div(className='right-left-side', style={'width': '50%', 'display': 'inline-block'}, children=[
                html.H3("Controls"),
                dcc.Slider(min=0, max=1, value=0.5, marks={0:'Very Bad', 0.25:'Bad', 0.5:'Neutral', 0.75:'Good', 1:'Very Good'}, tooltip={"placement": "bottom", "always_visible": True, 'template':'danceability'}, id='dance-slider'),
                dcc.Slider(min=0, max=1, value=0.5, marks={0:'Very Bad', 0.25:'Bad', 0.5:'Neutral', 0.75:'Good', 1:'Very Good'}, tooltip={"placement": "bottom", "always_visible": True, 'template':'energy'}, id='energy-slider'),
                dcc.Slider(min=0, max=1, value=0.5, marks={0:'Very Bad', 0.25:'Bad', 0.5:'Neutral', 0.75:'Good', 1:'Very Good'}, tooltip={"placement": "bottom", "always_visible": True, 'template':'preference'}, id='pref-slider'),
                html.Div(id='slider-output-container'),
                html.Div(id='checklist-output'),
                html.Div(id='output'),
                html.Button('Train the model!', id='train-btn', n_clicks=0), 
            ]),

            html.Div(className='right-right-side', style={'width': '50%', 'display': 'inline-block'}, children=[
                dcc.Graph(id='acc-vis', figure=fig2),  
            ]),
        ]),
    ]),

    dcc.Store(id='data-store'),
    dcc.Store(id='x_pool-store', data=X_pool),
    dcc.Store(id='y_pool-store', data=y_pool),
    dcc.Store(id='query-store', data=query_idx),
    dcc.Store(id='labeled-store', data=labeled_idx),    
    dcc.Store(id='pcp-store'),
    dcc.Store(id='reset-bool', data=False),
    dcc.Store(id='test-data', data=0),
], style={"display": "flex"})

# callback to update DataTable and visualization based on selections
@app.callback(
    Output('main-vis', 'figure'),
    Input('query-store', 'data'),
    Input('labeled-store', 'data'),
    State('checklist-output', 'children'),
)
def update_plot(query_idx, labeled_idx, current_children):
    print("ENTERED UPDATE PLOT")
    print("CURRENT CHILDREN BEFORE IN PLOT", current_children)
    x_data = data[~data.index.isin(labeled_idx)]['x_coor']
    y_data = data[~data.index.isin(labeled_idx)]['y_coor']
    artist_data = data[~data.index.isin(labeled_idx)]['artist']
    track_data = data[~data.index.isin(labeled_idx)]['track']
    release_date_data = data[~data.index.isin(labeled_idx)]['release_date']

    scatter_trace = go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        hoverinfo='text',
        text=[f'Artist: {a}<br>Track: {t}<br>Release Date: {d}' for a, t, d in zip(artist_data, track_data, release_date_data)],
        marker=dict(
            size=5
        )
    )

    # Creating layout options (if needed)
    layout = go.Layout(width=600, height=600, showlegend=False,)

    # Creating the figure
    fig = go.Figure(data=[scatter_trace], layout=layout)

    # TODO: 
    # - clickData does not work twice on the same item (for deselecting)
    # - remove item from pcp when unchecked
    # - change PCP to parallel categories
    # - other functionality for when there is no model fitted, perhaps randomly sample
    if labeled_idx[:-0 or None]:
        label_idx = labeled_idx[:-0 or None]
        x_data = data.iloc[label_idx]['x_coor']
        y_data = data.iloc[label_idx]['y_coor']
        artist_data = data.iloc[label_idx]['artist']
        track_data = data.iloc[label_idx]['track']
        release_date_data = data.iloc[label_idx]['release_date']

        scatter_trace = go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            hoverinfo='text',
            text=[f'Artist: {a}<br>Track: {t}<br>Release Date: {d}' for a, t, d in zip(artist_data, track_data, release_date_data)],
            marker=dict(
                color='red',
            )
        )
        fig.add_trace(scatter_trace)

    if query_idx:
        glyph_fig = plot_star_glyphs_from_dataframe(data.iloc[query_idx])
        for trace in glyph_fig.data:
            fig.add_trace(trace)

    fig.update_layout(clickmode='event+select', margin=dict(l=20, r=20, t=20, b=20))
    print("CURRENT CHILDREN AFTER IN PLOT", current_children)
    return fig


# TODO:
# - correctly work on button click (add red dots, remove from checkbox list)
@callback(
    Output('checklist-output', 'children', allow_duplicate=True),
    Output('pcp-vis', 'figure', allow_duplicate=True),
    Input('main-vis', 'selectedData'),
    State('checklist-output', 'children'),
    prevent_initial_call='initial_duplicate'
)
def lasso_select(selected_data, current_children):
    print("ENTERED LASSO SELECT")
    print("CURRENT CHILDREN BEFORE IN LASSO", current_children)
    if selected_data is None:
        print("CURRENT CHILDREN AFTER IN LASSO 1", current_children)
        return no_update
    
    points = selected_data.get('points', [])
    if not points:
        print("CURRENT CHILDREN AFTER IN LASSO 2", current_children)
        return no_update
    
    lasso = selected_data.get('lassoPoints', [])
    if not lasso:
        print("CURRENT CHILDREN AFTER IN LASSO 3", current_children)
        return no_update
    
    model_fitted = True
    for idx, committee in enumerate(comm_list):
        if not are_all_learners_fitted(committee):
            model_fitted = False

    df_al = pd.DataFrame(columns=data.columns)
    filtered_points = [point for point in selected_data['points'] if point['curveNumber'] == 0]
    for i in filtered_points:
        artist, track, rl = string_to_var(i['text'])
        filtered_data = data[(data['artist'] == artist) & (data['track'] == track)].dropna(axis=1, how='all')
        df_al_filtered = df_al.dropna(axis=1, how='all')
        df_al = pd.concat([df_al_filtered, filtered_data], ignore_index=True)
        # df_al = pd.concat([df_al, data[(data['artist'] == artist) & (data['track'] == track)]], ignore_index=True)

    x_df = df_al[NUM_FEATURES].to_numpy()
    std_list = np.zeros(len(x_df))

    if model_fitted:
        for comm_idx in range(n_comm):
            _, stds = comm_list[comm_idx].predict(x_df, return_std=True)
            std_list = [x + y for x, y in zip(std_list, stds)]

        # Convert std_list to a numpy array for efficient indexing
        std_array = np.array(std_list)

        # Number of elements to find
        n_largest = 5

        # Handle case where std_list has fewer than n_largest elements
        if len(std_array) <= n_largest:
            sorted_largest_indices = np.argsort(-std_array)
        else:
            # Get the indices of the n_largest elements
            largest_indices = np.argpartition(std_array, -n_largest)[-n_largest:]
            # Sort these indices by their corresponding values in descending order
            sorted_largest_indices = largest_indices[np.argsort(-std_array[largest_indices])]

        # Convert numpy array to list if needed
        query_idx = sorted_largest_indices.tolist()

    if not model_fitted:
        query_idx = np.random.randint(0, len(df_al), size=5)
        query_idx = list(set(query_idx))

    merged_df = data.merge(df_al.iloc[query_idx], on=COMMON_FEATURES, how='outer', indicator=True)

    # Filter rows that are in both DataFrames
    matching_rows = merged_df[merged_df['_merge'] == 'both']

    # Drop the indicator column if not needed
    matching_rows = matching_rows.drop(columns=['_merge'])
    samples = matching_rows[['artist_y', 'track_y', 'release_date']]

    if current_children is None:
        current_children = []

    df_pcp = pd.DataFrame(columns=data.columns)

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

        # df_pcp = pd.concat([df_pcp, data[(data['artist'] == artist) & (data['track'] == track)]], ignore_index=True)
        filtered_data = data[(data['artist'] == artist) & (data['track'] == track)].dropna(axis=1, how='all')
        df_pcp_filtered = df_pcp.dropna(axis=1, how='all')
        df_pcp = pd.concat([df_pcp_filtered, filtered_data], ignore_index=True)


    pcp = go.Figure(data=
        go.Parcoords(dimensions=[dict(range=[0, 1], label=feature, values=df_pcp[feature]) for feature in TEST_FEATURES])
    )
    pcp.update_layout(
        height=300,  # specify the height
        width=800    # specify the width
    )
    pcp.update_traces(
        dimensions=[
            {**d, **{"tickvals": np.linspace(0, 1, 5)}}
            for d in pcp.to_dict()["data"][0]["dimensions"]
        ],
        labelangle=-20
    )

    print("CURRENT CHILDREN AFTER IN LASSO 4", current_children)
    return current_children, pcp

@callback(
    Output('acc-vis', 'figure'),
    Output('x_pool-store', 'data'),
    Output('y_pool-store', 'data'),
    Output('query-store', 'data'), 
    Output('reset-bool', 'data', allow_duplicate=True),
    Input('train-btn', 'n_clicks'),
    State('x_pool-store', 'data'),
    State('y_pool-store', 'data'),
    State('pcp-store', 'data'),
    State('dance-slider', 'value'),
    State('energy-slider', 'value'),
    State('pref-slider', 'value'),
    prevent_initial_call='initial_duplicate'
)
def train_model(btn1, X_pool, y_pool, df, danceability, energy, preference):
    print("ENTERED TRAIN MODEL")
    if isinstance(X_pool, list):
        X_pool = np.array(X_pool)

    query_idx = []
    reset_bool = False
    df_perf = pd.DataFrame(np.transpose(performance_history), columns=['danceability', 'energy', 'preference'])
    fig2 = px.line(df_perf)
    fig2.update_layout(
        title="Model progress",
        xaxis_title="Training iteration",
        yaxis_title="Total standard deviation",
        legend_title="Features",
    )

    fig2.update_layout(
        height=300,  # specify the height
        width=350    # specify the width
    )

    y_pool_test = [danceability, energy, preference]

    if not df:
        return fig2, X_pool, y_pool, query_idx, reset_bool

    # if the button is clicked, train the model
    if "train-btn" == ctx.triggered_id:
        artist = pd.DataFrame(df)['artist'].tolist()
        track = pd.DataFrame(df)['track'].tolist()
        indices = data[(data['artist'].isin(artist)) & (data['track'].isin(track))].index.tolist()
        # query by committee
        std_list = np.zeros(len(X_pool))

        for comm_idx in range(n_comm):
            comm_list[comm_idx].teach(X_pool[indices], np.array([y_pool_test[comm_idx]] * X_pool[indices].shape[0]))

        for comm_idx in range(n_comm):
            _, stds = comm_list[comm_idx].predict(X_pool, return_std=True)
            std_list = [x + y for x, y in zip(std_list, stds)]
            performance_history[comm_idx] = np.append(performance_history[comm_idx], sum(stds))
            
        # Convert std_list to a numpy array for efficient indexing
        std_array = np.array(std_list)

        # Number of elements to find
        n_largest = 5

        # Handle case where std_list has fewer than n_largest elements
        if len(std_array) <= n_largest:
            sorted_largest_indices = np.argsort(-std_array)
        else:
            # Get the indices of the n_largest elements
            largest_indices = np.argpartition(std_array, -n_largest)[-n_largest:]
            # Sort these indices by their corresponding values in descending order
            sorted_largest_indices = largest_indices[np.argsort(-std_array[largest_indices])]

        # Convert numpy array to list if needed
        query_idx = sorted_largest_indices.tolist()
        
        # show the MSE performance over time
        df_perf = pd.DataFrame(np.transpose(performance_history), columns=['danceability', 'energy', 'preference'])
        fig2 = px.line(df_perf)
        fig2.update_layout(
            title="Model progress",
            xaxis_title="Training iteration",
            yaxis_title="Total standard deviation",
            legend_title="Features",
        )

        fig2.update_layout(
            height=300,  # specify the height
            width=350    # specify the width
        )

        reset_bool = True

    return fig2, X_pool, y_pool, query_idx, reset_bool

@app.callback(
    Output('checklist-output', 'children', allow_duplicate=True),
    Output('pcp-vis', 'figure', allow_duplicate=True),
    Output('pcp-store', 'data', allow_duplicate=True),
    Output('reset-bool', 'data', allow_duplicate=True), 
    Output('labeled-store', 'data', allow_duplicate=True),
    Input('main-vis', 'clickData'),
    Input('reset-bool', 'data'),
    Input('labeled-store', 'data'),
    State('pcp-store', 'data'),    
    State('query-store', 'data'),
    Input('checklist-output', 'children'),
    prevent_initial_call='initial_duplicate'
)
def handle_labeling(click_data, reset_bool, labeled_idx, pcp_df, query_idx, current_children):
    print('ENTERED HANDLE LABELING')
    print("CURRENT CHILDREN BEFORE IN HANDLE", current_children)
    if not click_data:
        MT = np.zeros(len(data.columns))
        df_pcp = pd.DataFrame([MT], columns=data.columns)
        pcp = go.Figure(data=
            go.Parcoords(dimensions=[dict(range=[0, 1], label=feature, values=df_pcp[feature]) for feature in TEST_FEATURES])
        )
        pcp.update_layout(
            height=300,  # specify the height
            width=800    # specify the width
        )
        pcp.update_traces(
            dimensions=[
                {**d, **{"tickvals": np.linspace(0, 1, 5)}}
                for d in pcp.to_dict()["data"][0]["dimensions"]
            ],
            labelangle=-20
        )
        print("CURRENT CHILDREN AFTER IN HANDLE 1", current_children)
        return current_children, pcp, pcp_df, reset_bool, labeled_idx
    
    if reset_bool:
        pcp_df = pd.DataFrame(pcp_df)
        reset_bool = False

        # if checkbox is checked, the item gets a 'value'
        value = [item['props']['value'] for item in current_children]
        
        # remove all items that are checked from the dataframe
        for i in value:
            if i:
                artist, track, rl = string_to_var(i[0])
                pcp_df = pcp_df[(pcp_df.artist != artist) & (pcp_df.track != track)]

                labeled_idx = labeled_idx + data[(data['artist'] == artist) & (data['track'] == track)].index.tolist()

        # create a new figure for the unchecked data
        pcp = go.Figure(data=
            go.Parcoords(dimensions=[dict(range=[0, 1], label=feature, values=pcp_df[feature]) for feature in TEST_FEATURES])
        )
        pcp.update_layout(
            height=300,  # specify the height
            width=800    # specify the width
        )
        pcp.update_traces(
            dimensions=[
                {**d, **{"tickvals": np.linspace(0, 1, 5)}}
                for d in pcp.to_dict()["data"][0]["dimensions"]
            ],
            labelangle=-20
        )

        # keep only the children that are not checked
        new_children = [i for i in current_children if not i['props']['value']]
        print("CURRENT CHILDREN AFTER IN HANDLE 2", current_children)
        return new_children, pcp, pcp_df.to_dict('records'), reset_bool, labeled_idx
    
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

    elif click_data['points'][0]['curveNumber'] > 1:
        id = (click_data['points'][0]['curveNumber'] - 3) // 2
        queried = query_idx[id]
        artist = data.iloc[queried]['artist']
        track = data.iloc[queried]['track']
        release_date = data.iloc[queried]['release_date']
        sample = var_to_string(artist, track, release_date)

    vis_df = data[(data['artist'].isin([artist])) & (data['track'].isin([track]))]

    if current_children is None:
        current_children = []

    if pcp_df is None:
        pcp_df = pd.DataFrame(columns=vis_df.columns)

    # Extract selected labels from current checklist items
    selected_labels = [item['props']['options'][0]['label'] for item in current_children]

    # Check if the clicked label is already selected
    if format_string(sample) in selected_labels:
        # If clicked label is selected, remove it from the checklist items
        new_children = [item for item in current_children if item['props']['options'][0]['label'] != format_string(sample)]
    else:
        # If clicked label is not selected, add it to the checklist items
        new_item = dcc.Checklist(
            id={'type': 'data-checklist', 'label': sample},
            options=[{'label': format_string(sample), 'value': sample}],
            value=[sample],
            inline=True
        )
        new_children = current_children + [new_item]

    pcp_df = pd.DataFrame(pcp_df)

    # Check conditions and create the result DataFrame
    if pcp_df.empty and not vis_df.empty:
        result = vis_df.copy()  # If a is empty but b has rows, result is b
    elif pcp_df.equals(vis_df):
        result = pd.DataFrame()  # If a and b are the same, result is an empty DataFrame
    else:
        # Find rows that are different between a and b
        diff_rows = pd.concat([pcp_df, vis_df]).drop_duplicates(keep=False)
        result = diff_rows

    pcp_df = result

    new_pcp = go.Figure(data=
        go.Parcoords(dimensions=[dict(range=[0, 1], label=feature, values=pcp_df[feature]) for feature in TEST_FEATURES])
    )
    new_pcp.update_traces(
    dimensions=[
        {**d, **{"tickvals": np.linspace(0, 1, 5)}}
        for d in new_pcp.to_dict()["data"][0]["dimensions"]
        ],
        labelangle=-20
    )

    print("CURRENT CHILDREN AFTER IN HANDLE 3", current_children)
    return new_children, new_pcp, pcp_df.to_dict('records'), reset_bool, labeled_idx

if __name__ == '__main__':
    app.run(debug=True)