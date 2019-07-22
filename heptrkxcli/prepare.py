# System
import os
import logging
import glob
from multiprocessing import Pool

# Externals
import numpy as np
import pandas as pd
import trackml.dataset


# Locals
from heptrkxcli.hitgraph import save_graph
from heptrkxcli.config import config_file
from heptrkxcli.events import events
from heptrkxcli.visualize_track import export_track

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

    cr = np.sqrt(cx**2 + cy**2)
    cphi = np.arctan2(cy, cx)

    return (cx, cy, radius)
 

def filter_segments(hits1, hits2, phi_slope_max, z0_max):
    """
    Construct a list of selected segments from the pairings
    between hits1 and hits2, filtered with the specified
    phi slope and z0 criteria.

    Returns: pd DataFrame of (index_1, index_2), corresponding to the
    DataFrame hit label-indices in hits1 and hits2, respectively.
    """
    # Start with all possible pairs of hits
    keys = ['r', 'phi', 'z', 'count', 'theta','particle_id']
    a = hits1[keys].assign(temp=0).reset_index()
    b = hits2[keys].assign(temp=0).reset_index()
    hit_pairs = a.merge(b, on='temp', suffixes=('_1', '_2')).drop(columns=['temp'])
    
    def calc_dphi(phi1, phi2):
        dphi = phi2 - phi1
        dphi[dphi > np.pi] -= 2*np.pi
        dphi[dphi < -np.pi] += 2*np.pi
        return dphi

    # Compute line through the points
    dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
    dtheta = calc_dphi(hit_pairs.theta_1, hit_pairs.theta_2)
    dz = hit_pairs.z_2 - hit_pairs.z_1
    dr = hit_pairs.r_2 - hit_pairs.r_1
    phi_slope = dphi / dr
    z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr
    scattering = 
    
    hit_pairs = hit_pairs.assign(
        t = (hit_pairs['particle_id_1'] == hit_pairs['particle_id_2'])*1,
        dphi = dphi,
        count = hit_pairs['count_1'],
        scattering = 
        dtheta = dtheta,
        dr = dr,
        dz = dz
    )

    # Filter segments according to criteria
    good_seg_mask = (phi_slope.abs() < phi_slope_max) & (z0.abs() < z0_max)
    return hit_pairs[['index_1', 'index_2', 't','dr','count','dphi','dtheta','dz']][good_seg_mask]

def select_hits(hits, truth, particles, sensors=None, pt_min=0):
    """
    We are not utilizing the entire dataset... we first want to
    remove noise as well as remove particles outside of our region
    of interest. This uses momentum truth values, so it is not representative
    of the true dataset. This results in a simplified dataset
    """

    # Select only hits from the specified sensors.
    if sensors != None:
        hits = hits.set_index(['volume_id', 'layer_id'])
        hits = hits.loc[hits.index.isin(sensors)]
        hits = hits.reset_index()

    # Apply trasverse momentum filter.
    if pt_min != 0:
        pt = np.sqrt(particles.px ** 2 + particles.py ** 2)
        particles = particles[pt > pt_min]

    # Calculate derived hits variables
    r = np.sqrt(hits.x**2 + hits.y**2)
    theta = np.arctan2(hits.z, hits.x)
    phi = np.arctan2(hits.y, hits.x)
    
    
    # Select the data columns we need
    hits = (hits.assign(r=r, phi=phi, theta=theta)
            .merge(truth[['hit_id', 'particle_id']], on='hit_id'))


    hits_count = hits['particle_id'].value_counts().drop(0).reset_index()
    hits_count.columns = ['particle_id', 'count']
    hits = hits.merge(hits_count, how='left', on="particle_id").fillna(1)

    # Remove duplicate hits
    hits = hits.set_index(['particle_id', 'volume_id', 'layer_id'], drop=False)
    hits = hits.loc[~hits.index.duplicated(keep='first')]
    hits.reset_index(drop=True, inplace=True)

    return hits

def section_iterator(hits, config):
    """Split hits according to provided phi and theta boundaries."""

    # Parse configuration
    phi_sections = config['n_phi_sections']
    theta_sections = config['n_theta_sections']

    # Calculate region
    phi_size = 2 * np.pi / phi_sections
    theta_size = 2 * np.pi / theta_sections
    hits = hits.assign(phi_region=lambda hit: np.floor((hit.phi + np.pi) / phi_size))
    hits = hits.assign(theta_region=lambda hit: np.floor((hit.theta + np.pi) / theta_size))

    for (phi_region, theta_region), group in hits.groupby(['phi_region', 'theta_region']):
        index = theta_region * phi_sections + phi_region
        if phi_region < phi_sections and theta_region < theta_sections:
            yield (index, group.reset_index(drop=True))

def choose_truth(hits):
    
    hits = hits.assign(t = lambda hit: np.sqrt(hit.x**2 + hit.y**2 + hit.z**2))
    hits = hits.sort_values(by=['particle_id','t'])
    hits = hits.assign(
            x0=hits['x'].shift(1),
            x2=hits['x'].shift(-1),
            y0=hits['y'].shift(1),
            y2=hits['y'].shift(-1))

    def mask(x):
        result = np.ones_like(x)
        result[0] = 0
        result[x.shape[0] - 1] = 0
        return result

    # Create a mask that selects all but the first and last element from each group
    mask = hits.groupby(['particle_id'])['particle_id'].transform(mask).astype(bool)

    hits = hits.loc[mask]

    # Applies mask to remove first and last hits from each particle and
    # calculates helix parameters for remaining hits
    hits = hits.assign(
            thcx = lambda n: fit_circle((n.x0,n.y0),(n.x,n.y),(n.x2,n.y2))[0],
            thcy = lambda n: fit_circle((n.x0,n.y0),(n.x,n.y),(n.x2,n.y2))[1],
            thr  = lambda n: fit_circle((n.x0,n.y0),(n.x,n.y),(n.x2,n.y2))[2])

    return hits

def normalize_nodes(hits, config):
    phi_mean = hits['phi'].mean()
    theta_mean = hits['theta'].mean()
    hits = hits.assign(phi = hits.phi - phi_mean, theta = hits.theta - theta_mean)
    phi_sections = config['n_phi_sections']
    theta_sections = config['n_theta_sections']
    return hits[['r', 'phi', 'theta', 'z']] / [1000, 2 * np.pi / phi_sections, 2 * np.pi / theta_sections, 1000]

def choose_edges(hits, layer_pairs, config):
    '''
    This function chooses segments
    '''
    phi_slope_max = config['phi_slope_max']
    z0_max = config['z0_max']
    layer_groups = hits.groupby(['volume_id', 'layer_id'])
    segments = []

    for (layer1, layer2) in layer_pairs:
        if all(x in layer_groups.groups.keys() for x in [layer1, layer2]):
            hits1 = layer_groups.get_group(layer1)
            hits2 = layer_groups.get_group(layer2)
            segments.append(filter_segments(hits1, hits2, phi_slope_max, z0_max))
        else:
            logging.info('skipping empty layer')

    return pd.concat(segments)

def compute_features(nodes, edges):
    for (index, node) in nodes.iterrows():
        in_nodes = edges[edges['index_2'] == index]['index_1'].values
        out_nodes = edges[edges['index_1'] == index]['index_2'].values
        for iindex in in_nodes:
            for oindex in out_nodes:
                in_node = nodes.iloc[iindex]
                out_node = nodes.iloc[oindex]
                
    
    return (nodes, edges)

def process_event(prefix, config):
    event_id = int(prefix[-9:])

    # Parse config
    output_dir = config['outdir']
    selection_config = config['selection']
    pt_min = selection_config['pt_min']
   
    # Load data
    logging.info('event %i: loading data' % event_id)
    hits = trackml.dataset.load_event_hits(prefix)
    particles = trackml.dataset.load_event_hits(prefix)
    truth = trackml.dataset.load_event_truth(prefix)

    # Select hits
    logging.info('event %i: selecting hits' % event_id)
    sensors = [(8,2), (8,4), (8,6), (8,8), (13,2), (13,4), (13,6), (13,8), (17,2), (17,4)]
    sensor_pairs = [[sensors[i], sensors[i + 1]] for i in range(0,9)]
    hits = select_hits(hits, truth, particles, sensors)
    
    # Construct the graph
    logging.info('event %i: constructing graphs' % event_id)
    base_prefix = os.path.basename(prefix)
 
    # Iterate through each section
    for index, nodes in section_iterator(hits, selection_config):
        # Compute data to be saved
        ids   = nodes['hit_id']
        edges = choose_edges(nodes, sensor_pairs, selection_config)
        #truth = choose_truth(nodes)
        nodes = normalize_nodes(nodes, selection_config)
        # Save graph to file
        input_path = os.path.join(os.path.join(output_dir, 'input'), '%s_g%03i' % (base_prefix, index))
        save_graph(input_path, {
            'nodes': nodes.to_numpy(),
            'edges': edges[['dr','dphi','dtheta','dz']].to_numpy(),
            'senders': edges['index_1'].to_numpy(),
            'receivers': edges['index_2'].to_numpy()
        })
        target_path = os.path.join(os.path.join(output_dir, 'target'), '%s_g%03i' % (base_prefix, index))
        save_graph(target_path, {
            'nodes': nodes.to_numpy(),
            'edges': edges[['t','count']].to_numpy(),
            'senders': edges['index_1'].to_numpy(),
            'receivers': edges['index_2'].to_numpy()
        })
def prepare():
    """Main function"""

    # Setup logging
    log_format = '%(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logging.info('Initializing')

    # Load configuration
    config = config_file()['preprocessing']
    n_events = config['n_events']
    outdir = config['outdir']
    outdir_input = os.path.join(config['outdir'],'input')
    outdir_target = os.path.join(config['outdir'],'target')

    file_prefixes = events()
    file_prefixes.sort()
    file_prefixes = file_prefixes[:n_events]

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_input, exist_ok=True)
    os.makedirs(outdir_target, exist_ok=True)

    logging.info('Writing outputs to ' + outdir)

    pool = Pool(20)
    pool.starmap(process_event, zip(file_prefixes, [config] * n_events), chunksize=10)

    logging.info('All done!')