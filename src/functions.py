import random
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.base import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px

# one hot encode the categorical columns
def one_hot_encode(X, column):
    enc = OneHotEncoder(handle_unknown='ignore')
    columns = sorted(X[column].unique())
    enc_df1 = pd.DataFrame(enc.fit_transform(X[[column]]).toarray(), columns=columns)
    X = X.join(enc_df1)
    X = X.drop([column], axis=1)
    return X

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
def get_pred_dm(X, comm_list, n_comm, lands):
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
def get_dm_coords(dm, dm_pred, lands):
    # dm_copy = dm + 0.75 * dm_pred
    dm_copy = dm[lands,:] + 0.75 * dm_pred[lands,:]
    xl_2 = landmark_MDS(dm_copy,lands,2)
    return xl_2[:,0], xl_2[:,1]

# remove non values from a nested list
def remove_none_values(nested_list):
    # Temporary list to store lists without None values
    cleaned_list = []

    # Iterate through each sublist in the nested list
    for sublist in nested_list:
        # Filter out None values and append the cleaned sublist to cleaned_list
        cleaned_sublist = [item for item in sublist if item is not None]
        cleaned_list.append(cleaned_sublist)

    return cleaned_list

# create the labeling density plot
def create_density_plot(perf_hist):
    name = ['danceability', 'energy', 'preference']
    colors = ['pink', 'goldenrod', 'green']

    data = remove_none_values(perf_hist)

    # Create the Plotly figure
    fig = go.Figure()

    # Loop over each dataset and compute the KDE using seaborn
    for idx, row in enumerate(data):
        sns_kde = sns.kdeplot(np.array(row), bw=0.1)
        line = sns_kde.get_lines()[0]
        x_data = line.get_xdata()
        y_data = line.get_ydata()

        # Add a trace for each dataset in Plotly
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', line_shape='spline', line=dict(color=colors[idx]), name=name[idx]))

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

def add_custom_legend(fig): 
    # define colors
    colors = ['blue', 'red', 'black', 'pink', 'goldenrod', 'green']
    
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

# initialize pcp attributes
bins = [-0.1, 0.2, 0.4, 0.6, 0.8, 1]
labels = ['Very bad', 'Bad', 'Neutral', 'Good', 'Very Good']
rl_labels = ['Very old', 'Old', 'Neutral', 'New', 'Very new']

def create_pcp(pcp_df, bins=bins, labels=labels, rl_labels=rl_labels):
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
    
# Function to create bar chart glyphs
def plot_bar_chart_glyphs_from_dataframe(data, n_comm, comm_list, TRAIN_FEATURES):
    # retrieve the data
    x_data = data['x_coor']
    y_data = data['y_coor']
    artist_data = data['artist']
    track_data = data['track']
    release_date_data = data['release_date']

    fig = go.Figure()

    # Create an initial scatter plot
    scatter_trace = go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        hoverinfo='text',
        text=[f'Artist: {a}<br>Track: {t}<br>Release Date: {d}' for a, t, d in zip(artist_data, track_data, release_date_data)],
        marker=dict(
            color='black',
        ),
        showlegend=False,
        )
    
    fig.add_trace(scatter_trace)

    # Scaling factor to increase the size of the glyphs
    scale_factor = 0.05

    offset_value = max(x_data) / 3

    # Define the offsets for the corners within each region
    offsets = {
        'top-left': (-offset_value, offset_value),
        'top-right': (offset_value, offset_value),
        'bottom-left': (-offset_value, -offset_value),
        'bottom-right': (offset_value, -offset_value)
    }

    # Determine the plot range (assuming default Plotly behavior if not specified)
    x_range = [data['x_coor'].min(), data['x_coor'].max()]
    y_range = [data['y_coor'].min(), data['y_coor'].max()]

    x_mid = (x_range[0] + x_range[1]) / 2
    y_mid = (y_range[0] + y_range[1]) / 2

    # Function to determine the corner based on the scatter point location
    def get_corner(x, y):
        if x < x_mid and y >= y_mid:
            return 'top-left'
        elif x >= x_mid and y >= y_mid:
            return 'top-right'
        elif x < x_mid and y < y_mid:
            return 'bottom-left'
        else:
            return 'bottom-right'

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
        colors = ['pink', 'goldenrod', 'green']

        # Get the corner for the current scatter point
        corner = get_corner(x, y)
        offset_x, offset_y = offsets[corner]

        # Calculate the new position for the glyph
        glyph_x = x + offset_x
        glyph_y = y + offset_y

        # Calculate x coordinates for the bars
        x_bars = np.linspace(-num_variables * (bar_width + bar_spacing) / 2 + bar_width / 2, 
                             num_variables * (bar_width + bar_spacing) / 2 - bar_width / 2, 
                             num_variables) + glyph_x
        
        # Add bars as individual traces
        for j in range(num_variables):
            fig.add_trace(
                go.Bar(
                    x=[x_bars[j]],
                    y=[values[j] * scale_factor],  # Scale bar height
                    width=[bar_width],
                    marker_color=colors[j],
                    base=glyph_y,
                    showlegend=False,
                    hoverinfo='none'  # Remove text inside bars
                )
            )

        # Calculate coordinates for the square box
        box_height = 1 * scale_factor  # Maximum height of the box, scaled
        total_width = num_variables * (bar_width + bar_spacing)  # Total width of all bars plus spacing
        box_x = [glyph_x - total_width / 2, glyph_x + total_width / 2, glyph_x + total_width / 2, glyph_x - total_width / 2, glyph_x - total_width / 2]
        box_y = [glyph_y, glyph_y, glyph_y + box_height, glyph_y + box_height, glyph_y]    

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

        # Determine line start and end based on glyph position and scatter position
        if glyph_y >= y_mid:  # Glyph in bottom half
            line_y_start = y
            line_y_end = (glyph_y - box_height / 2) + 0.5*scale_factor
        else:  # Glyph in top half
            line_y_start = y
            line_y_end = (glyph_y + box_height / 2) + 0.5*scale_factor

        # Add trace for the line connecting scatter to glyph
        fig.add_trace(
            go.Scatter(
                x=[x, glyph_x],
                y=[line_y_start, line_y_end],
                mode='lines',
                line=dict(width=0.5, color='black'),
                showlegend=False,
                hoverinfo='none'
            )
        )
        
    return fig