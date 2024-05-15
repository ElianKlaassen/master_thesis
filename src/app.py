from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx, State
import dash_bootstrap_components as dbc

import json
import math
import heapq
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from copy import deepcopy


from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

from modAL.models import ActiveLearner, CommitteeRegressor
from modAL.disagreement import max_std_sampling

df = pd.read_csv("src/data/tcc_ceds_music.csv")

scaler = MinMaxScaler()
norm_date = scaler.fit_transform(np.array(df['release_date']).reshape(-1, 1))
df['release_date'] = norm_date

# all features for the dimensionality reduction + model (temporary)
TEST_FEATURES = ['loudness', 'acousticness', 'instrumentalness', 'valence', 'blues', 'country', 'hip hop', 
                'jazz', 'pop', 'reggae', 'rock', 'feelings', 'music', 'night/time', 'obscene', 'romantic', 'sadness', 
                'violence', 'world/life']

ALL_FEATURES = ['loudness', 'acousticness', 'instrumentalness',
                'valence', 'blues', 'country', 'hip hop', 'jazz', 'pop', 'reggae',
                'rock', 'feelings', 'music', 'night/time', 'obscene', 'romantic',
                'sadness', 'violence', 'world/life', 'dance', 'ener', 'x_coor',
                'y_coor', 'artist', 'track', 'preference']

CHANGE_FEATURES = ['release_date', 'preference']

X = df[['release_date', 'genre', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'topic']]
Y = df[['danceability', 'energy']]

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
y_pool = [y_pool1, y_pool2]

# initializing Committee members
n_comm = 2
comm_list = list()
query_idx = []
labeled_idx = []
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
    Y_pred = committee.predict(X)
    return mean_squared_error(Y_pred, Y_true) 

# create a plot for the MSE over time
performance_history = [[0] for i in range(n_comm)]
for comm_idx in range(n_comm):
    _, stds = comm_list[comm_idx].predict(X_pool, return_std=True)
    performance_history[comm_idx] = [sum(stds)]

df_perf = pd.DataFrame(np.transpose(performance_history), columns=['danceability', 'energy'])
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
        values1 = row['loudness']
        values2 = row['acousticness']
        values3 = row['instrumentalness']
        values = [values1, values2, values3]

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
], style={"display": "flex"})

# TODO: fix lasso/box select
# - Glyph: use the correct features
# - clickData does not work twice on the same item (for deselecting)
# init commmittees without using labels
# - Make sure no y labels are used (except user given)
# - deselect selection box
# - when an item is deselected, it reappears when pressing the train button, since the 'clickdata' is still of this item somehow (probably due to children)
# callback to update DataTable and visualization based on selections
@app.callback(
    Output('main-vis', 'figure'),
    Output('data-store', 'data'),
    Input('main-vis', 'relayoutData'),
    Input('query-store', 'data'),
    Input('labeled-store', 'data'),
)
def update_data(relayout_data, query_idx, labeled_idx):
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

    # TODO: check consequences of [:-5 or None]
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

    return fig, json.dumps(data.to_dict("index"))

@callback(
    Output('acc-vis', 'figure'),
    Output('x_pool-store', 'data'),
    Output('y_pool-store', 'data'),
    Output('query-store', 'data'), 
    Output('labeled-store', 'data'),
    Output('reset-bool', 'data', allow_duplicate=True),
    Input('train-btn', 'n_clicks'),
    Input('x_pool-store', 'data'),
    Input('y_pool-store', 'data'),
    Input('labeled-store', 'data'),
    State('pcp-store', 'data'),
    State('dance-slider', 'value'),
    State('energy-slider', 'value'),
    State('pref-slider', 'value'),
    prevent_initial_call='initial_duplicate'
)
def displayClick(btn1, X_pool, y_pool, labeled_idx, df, danceability, energy, preference):
    if isinstance(X_pool, list):
        X_pool = np.array(X_pool)

    query_idx = []
    reset_bool = False
    df_perf = pd.DataFrame(np.transpose(performance_history), columns=['danceability', 'energy'])
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

    y_pool_test = [danceability, energy]

    if not df:
        return fig2, X_pool, y_pool, query_idx, labeled_idx, reset_bool

    # if the button is clicked, train the model
    if "train-btn" == ctx.triggered_id:
        artist = pd.DataFrame(df)['artist'].tolist()
        track = pd.DataFrame(df)['track'].tolist()
        indices = data[(data['artist'].isin(artist)) & (data['track'].isin(track))].index.tolist()
        # query by committee
        std_list = np.zeros(len(X_pool))

        for comm_idx in range(n_comm):
            _, stds = comm_list[comm_idx].predict(X_pool, return_std=True)
            std_list = [x + y for x, y in zip(std_list, stds)]
            performance_history[comm_idx] = np.append(performance_history[comm_idx], sum(stds))
            
        query_idx = [std_list.index(i) for i in heapq.nlargest(5, std_list)]

        for comm_idx in range(n_comm):
            comm_list[comm_idx].teach(X_pool[indices], np.array([y_pool_test[comm_idx]] * X_pool[indices].shape[0]))
        
        # show the MSE performance over time
        df_perf = pd.DataFrame(np.transpose(performance_history), columns=['danceability', 'energy'])
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

        labeled_idx = labeled_idx + indices

        reset_bool = True

    return fig2, X_pool, y_pool, query_idx, labeled_idx, reset_bool

@app.callback(
    Output('checklist-output', 'children'),
    Output('pcp-vis', 'figure'),
    Output('pcp-store', 'data'),
    Output('reset-bool', 'data', allow_duplicate=True), 
    Input('main-vis', 'clickData'),
    Input('reset-bool', 'data'),
    State('pcp-store', 'data'),    
    State('query-store', 'data'),
    State('labeled-store', 'data'),
    State('checklist-output', 'children'),
    prevent_initial_call='initial_duplicate'
)
def update_output_checklist(click_data, reset_bool, pcp_df, query_idx, labeled_idx, current_children):
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

    if not click_data:
        return current_children, pcp, pcp_df, reset_bool
    
    if reset_bool:
        reset_bool = False
        return [], pcp, df_pcp.to_dict('records'), reset_bool
    
    artist, track, release_date, sample = None, None, None, None
    if click_data['points'][0]['curveNumber'] == 0:
        sample = click_data['points'][0]['text']
        artist, track, release_date = string_to_var(sample)

    if click_data['points'][0]['curveNumber'] == 1:
        id = click_data['points'][0]['pointIndex']
        queried = labeled_idx[id]
        artist = data.iloc[queried]['artist']
        track = data.iloc[queried]['track']
        release_date = data.iloc[queried]['release_date']
        sample = var_to_string(artist, track, release_date)

    if click_data['points'][0]['curveNumber'] > 1:
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
            # options=[{'label': sample, 'value': sample}],
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

    return new_children, new_pcp, pcp_df.to_dict('records'), reset_bool

if __name__ == '__main__':
    app.run(debug=True)