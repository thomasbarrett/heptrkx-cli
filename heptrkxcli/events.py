import glob, os
from heptrkxcli.config import config_file

def events():
    '''
    Returns a list of all events in the train_all directory, excluding any
    hits, particles, or truth suffixes.
    '''
    train_all = config_file()['train_all']
    hit_pattern = os.path.join(train_all,'*-hits.csv')
    hits = glob.glob(hit_pattern)
    events = list(map(lambda hit: hit.split('-hits.csv')[0], sorted(hits)))
    return events

def display_event_summary():
    print('events: %i' % len(events()))