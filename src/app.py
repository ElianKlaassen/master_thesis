from dash import Dash, html, dcc, callback, Output, Input, ctx, State, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import MATCH, ALL

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from scipy.spatial.distance import pdist, squareform

from modAL.models import ActiveLearner, CommitteeRegressor
from modAL.disagreement import max_std_sampling

import pandas as pd
import numpy as np
from copy import deepcopy
from PIL import Image
import base64
from io import BytesIO
import seaborn as sns
from scipy.spatial import distance
import scipy as sp
import random

# TODO:
# - remove item from pcp when unchecked (if needed)
# - correctly approach the relabeling of samples

# load dataset, select only a subset 
df = pd.read_csv("src/data/tcc_ceds_music.csv")
df = df[df['release_date'] >= 1999]

# normalize the release date
scaler = MinMaxScaler()
norm_date = scaler.fit_transform(np.array(df['release_date']).reshape(-1, 1))
df['release_date'] = norm_date
df['preference'] = 0

# different feature arrays
TRAIN_FEATURES = ['release_date', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'blues', 'country', 'hip hop', 
                'jazz', 'pop', 'reggae', 'rock', 'feelings', 'music', 'night/time', 'obscene', 'romantic', 'sadness', 
                'violence', 'world/life']

PCP_FEATURES = ['release_date', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'genre', 'topic']

COMMON_FEATURES = ['release_date', 'loudness', 'acousticness', 'instrumentalness', 'valence']

# initialize pcp attributes
bins = [-0.1, 0.2, 0.4, 0.6, 0.8, 1]
labels = ['Very bad', 'Bad', 'Neutral', 'Good', 'Very Good']
rl_labels = ['Very old', 'Old', 'Neutral', 'New', 'Very new']

# remove duplicates
df = df.drop_duplicates(subset=COMMON_FEATURES).reset_index(drop=True)

# define X and Y frames
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

# check if a model if fitted
def is_model_fitted(model):
    try:
        # check if the model is fitted by checking one of the fitted attributes
        check_is_fitted(model)
        return True
    except NotFittedError:
        return False
    
# check if all learners are fitted 
def are_all_learners_fitted(committee):
    for learner in committee.learner_list:
        if not is_model_fitted(learner.estimator):
            return False
    return True

# taken from https://github.com/danilomotta/LMDS/tree/master
def landmark_MDS(D, lands, dim):
	Dl = D[:,lands]
	n = len(Dl)

	# Centering matrix
	H = - np.ones((n, n))/n
	np.fill_diagonal(H,1-1/n)
	# YY^T
	H = -H.dot(Dl**2).dot(H)/2

	# Diagonalize
	evals, evecs = np.linalg.eigh(H)

	# Sort by eigenvalue in descending order
	idx   = np.argsort(evals)[::-1]
	evals = evals[idx]
	evecs = evecs[:,idx]

	# Compute the coordinates using positive-eigenvalued components only
	w, = np.where(evals > 0)
	if dim:
		arr = evals
		w = arr.argsort()[-dim:][::-1]
		if np.any(evals[w]<0):
			print('Error: Not enough positive eigenvalues for the selected dim.')
			return []
	if w.size==0:
		print('Error: matrix is negative definite.')
		return []

	V = evecs[:,w]
	L = V.dot(np.diag(np.sqrt(evals[w]))).T
	N = D.shape[1]
	Lh = V.dot(np.diag(1./np.sqrt(evals[w]))).T
	Dm = D - np.tile(np.mean(Dl,axis=1),(N, 1)).T
	dim = w.size
	X = -Lh.dot(Dm)/2.
	X -= np.tile(np.mean(X,axis=1),(N, 1)).T

	_, evecs = sp.linalg.eigh(X.dot(X.T))

	return (evecs[:,::-1].T.dot(X)).T

# get distance matrix for the prediction variables
def get_pred_dm(X):
    model_fitted = True
    for idx, committee in enumerate(comm_list):
        if not are_all_learners_fitted(committee):
            model_fitted = False
            
    preds = []
    if model_fitted:
        for comm_idx in range(n_comm):
            pred = comm_list[comm_idx].predict(X.to_numpy())
            preds.append(pred)        
        result = np.array(preds).T
    else:
        result = np.full((X.shape[0], 3), 0.5)
        
    dm = pairwise_distances(result, result, metric='cosine', n_jobs=-1)
    
    return dm

# get the dimensions for the combination of distance matrices
def get_dm_coords(dm, dm_pred):
    dm = dm + 0.5 * dm_pred

    # normalize the matrix
    lands = random.sample(range(0,dm.shape[0],1),int(dm.shape[0]**0.5))
    lands = np.array(lands,dtype=int)

    dm_copy = dm[lands,:] + 0.5 * dm_pred[lands,:]
    xl_2 = landmark_MDS(dm_copy,lands,2)
    return xl_2[:,0], xl_2[:,1]

# create dimensionality reduction
dm1 = pairwise_distances(X.to_numpy()[:,:5], X.to_numpy()[:,:5], metric='cosine', n_jobs=-1)
jaccard_distances = pdist(X.to_numpy()[:,5:], metric='jaccard')
dm2 = squareform(jaccard_distances)

dm = dm1.copy()  # Make a copy of distance_matrix_1 to avoid modifying the original
# Perform in-place operations to optimize performance
np.multiply(dm, 0.375, out=dm)  # Scale dm by 0.375 in place
np.add(dm, 0.125 * dm2, out=dm)  # Add 0.125 * dm2 to dm in place

dm_pred = get_pred_dm(X)

X_dm, y_dm = get_dm_coords(dm, dm_pred)

# create new dataframe
data = X.copy()
data['x_coor'], data['y_coor'] = X_dm, y_dm
data['artist'], data['track'] = df[['artist_name']], df[['track_name']]
data['genre'], data['topic'] = df['genre'], df['topic']

# create training dataset
train = pd.concat([X, Y], axis=1)

# generate the pool
X_pool = deepcopy(train[TRAIN_FEATURES].to_numpy())
y_pool1 = deepcopy(train['danceability'].to_numpy())
y_pool2 = deepcopy(train['energy'].to_numpy())
y_pool3 = deepcopy(train['preference'].to_numpy())
y_pool = [y_pool1, y_pool2, y_pool3]

def create_density_plot(data):
    name = ['danceability', 'energy', 'preference']

    # Create the Plotly figure
    fig = go.Figure()

    # Loop over each dataset and compute the KDE using seaborn
    for idx, row in enumerate(data):
        sns_kde = sns.kdeplot(np.array(row), bw=0.1)
        line = sns_kde.get_lines()[0]
        x_data = line.get_xdata()
        y_data = line.get_ydata()

        # Add a trace for each dataset in Plotly
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', line_shape='spline', name=name[idx]))

        # Clear the seaborn plot
        sns_kde.clear()
        
    fig.update_yaxes(visible=False, showticklabels=False)
    
    # Define the custom tick labels
    custom_ticks = {0: 'Very Bad', 0.25: 'Bad', 0.5: 'Neutral', 0.75: 'Good', 1: 'Very Good'}

    # Update the x-axis with custom tick labels
    fig.update_xaxes(
        tickvals=list(custom_ticks.keys()),
        ticktext=list(custom_ticks.values()),
        range=[-0.25, 1.25],  # Set the range to cover the full span of the x-axis
    )
    return fig

# create a plot for the model performance over time
performance_history = [[-2, 2] for i in range(n_comm)]
fig2 = create_density_plot(performance_history)

def add_custom_legend(fig): 
    # define colors
    colors = ['blue', 'red', 'black', 'pink', 'yellow', 'green']
    
    # Add a dummy trace for the custom legend item 'unlabeled'
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color=colors[0]),
        legendgroup='Scatters',
        showlegend=True,
        name='Unlabeled'
    ))

    # Add a dummy trace for the custom legend item 'labeled'
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color=colors[1]),
        legendgroup='Scatters',
        showlegend=True,
        name='Labeled'
    ))
    
    # Add a dummy trace for the custom legend item 'Glyph'
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color=colors[2]),
        legendgroup='Glyph',
        showlegend=True,
        name='Glyph:'
    ))
    
    # Add a dummy trace for the custom legend item 'danceability'
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color=colors[3]),
        legendgroup='Glyph',
        showlegend=True,
        name='Danceability'
    ))
    
    # Add a dummy trace for the custom legend item 'energy'
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color=colors[4]),
        legendgroup='Glyph',
        showlegend=True,
        name='Energy'
    ))
    
    # Add a dummy trace for the custom legend item 'preference'
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color=colors[5]),
        legendgroup='Glyph',
        showlegend=True,
        name='Preference'
    ))
    
    return fig

# create a parallel categories plot
def create_pcp(pcp_df):
    pcp_df['Release Date'] = pd.cut(pcp_df['release_date'], bins=bins, labels=rl_labels)
    pcp_df['Loudness'] = pd.cut(pcp_df['loudness'], bins=bins, labels=labels)
    pcp_df['Acousticness'] = pd.cut(pcp_df['acousticness'], bins=bins, labels=labels)
    pcp_df['Instrumentalness'] = pd.cut(pcp_df['instrumentalness'], bins=bins, labels=labels)
    pcp_df['Valence'] = pd.cut(pcp_df['valence'], bins=bins, labels=labels)
    pcp = px.parallel_categories(pcp_df[['Release Date', 'Loudness', 'Acousticness', 'Instrumentalness', 'Valence', 'genre', 'topic']])
    
    pcp.update_layout(
        height=300,  # specify the height
        width=800    # specify the width
    )
    return pcp

# create processable string from variables
def var_to_string(artist, track, release_date):
    # construct the string with the provided information
    string = f"Artist: {artist}<br>Track: {track}<br>Release Date: {release_date}"    
    return string


# create variables from processable string
def string_to_var(string):
    # split the string into separate lines
    lines = string.split("<br>")

    # initialize variables
    artist = None
    track = None
    release_date = None

    # iterate over each line and extract information
    for line in lines:
        if line.startswith("Artist:"):
            artist = line.split("Artist: ")[1].strip()
        elif line.startswith("Track:"):
            track = line.split("Track: ")[1].strip()
        elif line.startswith("Release Date:"):
            release_date = float(line.split("Release Date: ")[1])

    return artist, track, release_date

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

    preds = get_pred(artist, track)
    for i in range(len(preds)):
        preds[i] = round(preds[i], 2)

    # construct the formatted string
    formatted_string = f"'{track}' by '{artist}'. Current prediction: {preds}"
    
    return formatted_string

def get_pred(artist, track):
    x = data[(data['artist'] == artist) & (data['track'] == track)][TRAIN_FEATURES]
    
    model_fitted = True
    for idx, committee in enumerate(comm_list):
        if not are_all_learners_fitted(committee):
            model_fitted = False
            
    preds = []
    if model_fitted:
        for comm_idx in range(n_comm):
            pred = comm_list[comm_idx].predict(x.to_numpy())
            preds.append(pred[0])
        return preds
        
    else:
        return [0, 0, 0]

# Function to create bar chart glyphs
def plot_bar_chart_glyphs_from_dataframe(data):
    # Create an initial scatter plot
    fig = go.Figure()

    # Scaling factor to increase the size of the glyphs
    scale_factor = 5

    # Add traces for each bar chart glyph
    for i, row in data.iterrows():
        x = row['x_coor']
        y = row['y_coor']
        values = [0 for _ in range(n_comm)]

        for comm_idx in range(n_comm):
            values[comm_idx] = comm_list[comm_idx].predict([row[TRAIN_FEATURES].tolist()])[0]

        num_variables = len(values)
        bar_width = 0.5 * scale_factor  # Width of each bar, scaled for larger glyphs
        bar_spacing = 0.05 * scale_factor  # Reduced spacing between bars
        # colors = px.colors.qualitative.Light24[:num_variables]  # Different colors for each bar
        colors = ['yellow', 'pink', 'green']

        # Calculate x coordinates for the bars
        x_bars = np.linspace(-num_variables * (bar_width + bar_spacing) / 2 + bar_width / 2, 
                             num_variables * (bar_width + bar_spacing) / 2 - bar_width / 2, 
                             num_variables) + x
        
        # Add bars as individual traces
        for j in range(num_variables):
            fig.add_trace(
                go.Bar(
                    x=[x_bars[j]],
                    y=[values[j] * scale_factor],  # Scale bar height
                    width=[bar_width],
                    marker_color=colors[j],
                    base=y,
                    showlegend=False,
                    hoverinfo='none'  # Remove text inside bars
                )
            )

        # Calculate coordinates for the square box
        box_height = 1 * scale_factor  # Maximum height of the box, scaled
        total_width = num_variables * (bar_width + bar_spacing)  # Total width of all bars plus spacing
        box_x = [x - total_width / 2, x + total_width / 2, x + total_width / 2, x - total_width / 2, x - total_width / 2]
        box_y = [y, y, y + box_height, y + box_height, y]    

        # Add trace for the square box
        fig.add_trace(
            go.Scatter(
                x=box_x,
                y=box_y,
                mode='lines',
                fill='toself', 
                fillcolor='rgba(0,0,0,0)', 
                line=dict(width=0.5, color='black'),  # black box line
                showlegend=False,
                hoverinfo='text',
                text=f"Artist: {row['artist']}<br>Track: {row['track']}<br>Release Date: {row['release_date']}"
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
                                width=750, height=600)
fig = add_custom_legend(fig)
fig.update_layout(clickmode='event+select', margin=dict(l=20, r=20, t=20, b=20),)

# create a mock-up pcp
MT = np.zeros(len(data.columns))
df_pcp = pd.DataFrame([MT], columns=data.columns)
pcp = create_pcp(df_pcp)

# adjust height and width for the performance plot
fig2.update_layout(
    height=300,  # specify the height
    width=350    # specify the width
)

# dashboard layout
app = Dash(__name__)
app.layout = html.Div([
    # Main visualization
    html.Div(className='left-side', style={'width': '50%', 'display': 'inline-block'}, children=[
        html.H2("Data visualization"),
        dbc.Col(dcc.Graph(id='main-vis', figure=fig)),
    ]),

    html.Div(className='right-side', style={'width': '50%', 'display': 'inline-block'}, children=[
        # PCP plot
        html.Div(className='top-right', children=[
            html.H3("Data Attributes"),
            dcc.Graph(id='pcp-vis', figure=pcp),
        ]),

        # slider, checkboxes, and button
        html.Div(className='bottom-right', children=[
            html.Div(className='right-left-side', style={'width': '50%', 'display': 'inline-block'}, children=[
                html.H3("Controls"),
                dcc.Slider(min=0, max=1, value=0.5, marks={0:'Very Bad', 0.25:'Bad', 0.5:'Neutral', 0.75:'Good', 1:'Very Good'}, tooltip={"placement": "bottom", "always_visible": True, 'template':'danceability'}, id='dance-slider'),
                dcc.Slider(min=0, max=1, value=0.5, marks={0:'Very Bad', 0.25:'Bad', 0.5:'Neutral', 0.75:'Good', 1:'Very Good'}, tooltip={"placement": "bottom", "always_visible": True, 'template':'energy'}, id='energy-slider'),
                dcc.Slider(min=0, max=1, value=0.5, marks={0:'Very Bad', 0.25:'Bad', 0.5:'Neutral', 0.75:'Good', 1:'Very Good'}, tooltip={"placement": "bottom", "always_visible": True, 'template':'preference'}, id='pref-slider'),
                html.Div(id='checklist-output'),
                html.Button('Train the model!', id='train-btn', n_clicks=0), 
                html.Button('Remove selected items', id='remove-btn', n_clicks=0),
            ]),

            # performance plot
            html.Div(className='right-right-side', style={'width': '50%', 'display': 'inline-block'}, children=[
                dcc.Graph(id='acc-vis', figure=fig2),  
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
], style={"display": "flex"})

# callback to update the scatters in the main visualization
@app.callback(
    Output('main-vis', 'figure'),
    Input('query-store', 'data'),
    Input('labeled-store', 'data'),
    State('checklist-output', 'children'),
    State('main-vis', 'relayoutData'),
)
def update_plot(query_idx, labeled_idx, current_children, relayout_data):
    blue_color_scale = [
        [0, '#add8e6'],  # Light blue
        [1, '#00008b']   # Dark blue
    ]

    # check if the AL models are fitted
    model_fitted = True
    for idx, committee in enumerate(comm_list):
        if not are_all_learners_fitted(committee):
            model_fitted = False

    preds = [1 for i in range(len(data))]
    # if the model is fitted, find the 5 most usefull points according to the AL model
    if model_fitted:
        preds, stds = comm_list[2].predict(X_pool, return_std=True)

    data_copy = data.copy()
    data_copy['preds'] = preds

    # retrieve the data
    preds_data = data_copy[~data_copy.index.isin(labeled_idx)]['preds']
    x_data = data_copy[~data_copy.index.isin(labeled_idx)]['x_coor']
    y_data = data_copy[~data_copy.index.isin(labeled_idx)]['y_coor']
    artist_data = data_copy[~data_copy.index.isin(labeled_idx)]['artist']
    track_data = data_copy[~data_copy.index.isin(labeled_idx)]['track']
    release_date_data = data_copy[~data_copy.index.isin(labeled_idx)]['release_date']

    # create the default scatter plot
    scatter_trace = go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        hoverinfo='text',
        text=[f'Artist: {a}<br>Track: {t}<br>Release Date: {d}' for a, t, d in zip(artist_data, track_data, release_date_data)],
        marker=dict(
            color=preds_data,
            colorscale=blue_color_scale,  # Use the custom blue color scale
            size=5
        ),
        showlegend=False,
        )
    layout = go.Layout(width=750, height=600)
    fig = go.Figure(data=[scatter_trace], layout=layout)

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
        fig.add_trace(scatter_trace)

    # create the scatter plot for the AL guidance scatters
    if query_idx:
        glyph_fig = plot_bar_chart_glyphs_from_dataframe(data.iloc[query_idx])
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

    fig = add_custom_legend(fig)

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

    fig.update_layout(
        plot_bgcolor='white'
    )
    return fig

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
    if isinstance(X_pool, list):
        X_pool = np.array(X_pool)

    # initialize variables
    query_idx = []
    reset_bool = False
    fig2 = create_density_plot(performance_history)
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

        # train the model on the new instances
        for comm_idx in range(n_comm):
            comm_list[comm_idx].teach(X_pool[indices], np.array([y_pool_test[comm_idx]] * X_pool[indices].shape[0]))

        # get new predictions and performance
        for comm_idx in range(n_comm):
            _, stds = comm_list[comm_idx].predict(X_pool, return_std=True)
            std_list = [x + y for x, y in zip(std_list, stds)]
            for i in range(0, len(pd.DataFrame(df))):
                performance_history[comm_idx] = np.append(performance_history[comm_idx], y_pool_test[comm_idx])
            
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
        fig2.update_layout(
            height=300,  # specify the height
            width=350    # specify the width
        )

        reset_bool = True

    return fig2, X_pool, y_pool, query_idx, reset_bool

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
    prevent_initial_call='initial_duplicate'
)
def handle_labeling(click_data, reset_bool, labeled_idx, pcp_df, query_idx, current_children):
    MT = np.zeros(len(data.columns))
    df_pcp = pd.DataFrame([MT], columns=data.columns)
    pcp = create_pcp(df_pcp)
    
    # if the button is pressed, remove check items, keep unchecked items
    new_children = current_children
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
        pcp = create_pcp(pcp_df)

        # keep only the children that are not checked
        new_children = [i for i in current_children if not i['props']['value']]
        #TODO maybe delete return statement
        return new_children, pcp, pcp_df.to_dict('records'), reset_bool, labeled_idx, None
    
    # if no data is clicked, return a mock-up pcp
    if not click_data:
        if not new_children:            
            if not isinstance(pcp_df, list) and pcp_df is not None:
                pcp_df = pcp_df.to_dict('records')
            return new_children, pcp, pcp_df, reset_bool, labeled_idx, None
        else:
            pcp_df = pd.DataFrame(columns=data.columns)
            selected_labels = [item['props']['options'][0]['value'] for item in new_children]
            for i in selected_labels:
                artist, track, rl = string_to_var(i)
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

    elif click_data['points'][0]['curveNumber'] > 1:
        id = (click_data['points'][0]['curveNumber'] - 5) // 4
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

    print('PCP 5', pcp_df)

    # create a new pcp with the resulting df
    new_pcp = create_pcp(pcp_df)

    return new_children, new_pcp, pcp_df.to_dict('records'), reset_bool, labeled_idx, None

if __name__ == '__main__':
    app.run(debug=True)