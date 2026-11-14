"""Published experiment recipes and their reporting endpoints."""
import json
from pathlib import Path

CONFIGS = Path(__file__).with_name('configs')
DATASETS = ('fashion_mnist', 'cifar100', 'office_home', 'retina')
NAMES = {
    'fedavg': 'FedAvg',
    'fedfv': 'FedFV',
    'fedmgda': 'FedMGDA+',
    'fedmdfg': 'FedMDFG-Agg',
    'fedld': 'FedLD-Agg',
    'fedawa': 'FedAWA-Agg',
}
MECHANISMS = ('baseline', 'sam', 'asam', 'gf', 'gloss')
SEEDS = (0, 1, 2)


def recipes(dataset):
    return json.loads((CONFIGS / f'{dataset}.json').read_text(encoding='utf-8'))


def recipe(dataset, coordinator, mechanism, seed):
    if seed not in SEEDS:
        raise ValueError(f'Paper seeds are {SEEDS}')
    try:
        value = recipes(dataset)[f'{coordinator}:{mechanism}']
    except KeyError as error:
        raise ValueError(f'Unavailable pairing: {coordinator}/{mechanism}') from error
    value['config']['seed'] = seed
    return value
