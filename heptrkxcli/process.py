from heptrkxcli.events import events
from trackml.dataset import load_event
import numpy as np
import pandas as pd


def fit_circle(p1, p2, p3):
    '''
    Fit a circle through the three given points
    Returns: The center and radius of the circle
    '''
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return (cx, cy, radius)
 
def process_particles(event):
    hits, cells, particles, truth = load_event(event)
   
    # select interesting columns and calculate distance from origin 't'
    hits = hits[['hit_id','x','y','z']]
    hits = hits.assign(t=lambda hit: np.sqrt(hit.x**2 + hit.y**2 + hit.z**2))

    # merge datasets to associate particle_id, hit_id, layer_id, and module_id,
    # and sort by distance from the origin
    data = pd.merge(truth[['particle_id','hit_id']], hits, on='hit_id', how='inner').sort_values(by=['particle_id','t'])

    # Add a new column to each row containing the previous and next hit
    data = data.assign(
            x0=data['x'].shift(1),
            x1=data['x'],
            x2=data['x'].shift(-1),
            y0=data['y'].shift(1),
            y1=data['y'],
            y2=data['y'].shift(-1))

    # a mask that selects all but the first and last rows
    def mask(x):
        result = np.ones_like(x)
        result[0] = 0
        result[x.shape[0] - 1] = 0
        return result

    # Create a mask that selects all but the first and last element from each group
    mask = data.groupby(['particle_id'])['particle_id'].transform(mask).astype(bool)

    # Applies mask to remove first and last hits from each particle and
    # calculates helix parameters for remaining hits
    data = data.loc[mask].assign(
            hx=lambda n: fit_circle((n.x0,n.y0),(n.x1,n.y1),(n.x2,n.y2))[0],
            hy=lambda n: fit_circle((n.x0,n.y0),(n.x1,n.y1),(n.x2,n.y2))[1],
            hr=lambda n: fit_circle((n.x0,n.y0),(n.x1,n.y1),(n.x2,n.y2))[2])


    # Iterate through each group and yield it
    for name, group in data.groupby('particle_id'):
        yield (name, group[['x','y','z','hx','hy','hr']])
