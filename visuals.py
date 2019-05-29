import plotly
import plotly.graph_objs as go
import pandas as pd
import numpy as np

plotly.offline.init_notebook_mode(connected=True)

def plot_trajectory_3d(csv_file, title, filename, width=600, height=600, axis_ranges=([-1,1], [-1,1], [-1,1])):

    df = pd.read_csv(csv_file)

    trace = go.Scatter3d(
        x=df['x'], y=df['y'], z=df['z'],
        marker=dict(
            size=4,
            color=df['time'],
            colorscale='Portland'
        ),
        line=dict(
            color='#000000',
            width=1
        )
    )

    data = [trace]

    layout = dict(
        title=title,
        width=width,
        height=height,
        autosize=False,
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 0, 0)',
                showbackground=True,
                backgroundcolor='rgb(220, 220, 220)',
                range=axis_ranges[0]
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 0, 0)',
                showbackground=True,
                backgroundcolor='rgb(220, 220, 220)',
                range=axis_ranges[1],
                color='green'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 0, 0)',
                showbackground=True,
                backgroundcolor='rgb(180, 180, 180)',
                range=axis_ranges[2],
                color='blue'
            ),
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2, y=2, z=0.1)
            )
        )
    )

    fig = dict(data=data, layout=layout)

    plotly.offline.iplot(fig, filename=filename, validate=True)