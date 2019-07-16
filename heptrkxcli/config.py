from os import path, makedirs, getenv
import yaml
import sys

def config_dir():
    home_path = path.expanduser('~')
    config_path = path.join(home_path, '.heptrkx')
    makedirs(config_path, exist_ok=True)
    return config_path

def config_file_path():
    ''' Return the configuration file path'''
    default_path = path.join(config_dir(), 'config.yaml')
    return getenv('HEPTRKX_CLI_CONFIG',default_path)

def config_file_exists():
    ''' Return True if configuration file exists and False otherwise'''
    return path.exists(config_file_path())

def config_file():
    '''
    Returns the contents of the configuration file.

    If no configuration file exists, a new one will be created at the location
    ~/.heptrkx/config.yaml with a set pf default configurations that work on
    gpuservers
    '''
    if config_file.cache != None:
        return config_file.cache

    if config_file_exists():
        with open(config_file_path(), 'r') as ifile:
            config_file.cache = yaml.load(ifile, Loader=yaml.FullLoader)
    else:
        config_file.cache = dict(
            train_all = '/bigdata/shared/TrackML/train_all',
        )
        with open(config_file_path(), 'w') as ofile:
            yaml.dump(config_file.cache, ofile)

    return config_file.cache
    

config_file.cache = None