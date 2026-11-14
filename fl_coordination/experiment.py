"""Dispatch to the published numerical adapters and select the reporting endpoint."""
import hashlib
import json
import math
from importlib import import_module
from pathlib import Path

from .recipes import CONFIGS, NAMES, recipe

PACKAGE = Path(__file__).parent
IMPL = 'fl_coordination._impl'


def module(name):
    return import_module(f'{IMPL}.{name}')


def source_manifest():
    paths = sorted([*PACKAGE.rglob('*.py'), *CONFIGS.glob('*.json')])
    files = {p.relative_to(PACKAGE).as_posix():
             hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
             for p in paths}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'sha256': digest, 'files': files}


def load_data(dataset, data_root, config):
    if dataset in ('fashion_mnist', 'cifar100'):
        reader, provider = ('fedseries', 'fedteam') if dataset == 'fashion_mnist' else (
            'fedcifar', 'cifar_recipes')
        module(provider + '.data').ensure(data_root)
        return module(reader + '.data').Data.load(data_root, config, False)
    reader = 'fedoffice' if dataset == 'office_home' else 'fedretina'
    if dataset == 'retina' and (Path(data_root)/'unpacked'/'Retina'/'labels.csv').is_file():
        data_root = Path(data_root)/'unpacked'/'Retina'
    return module(reader + '.data').Data(data_root, config)


def location(root, dataset, seed, coordinator, mechanism):
    return Path(root).resolve() / dataset / f'seed_{seed}' / coordinator / mechanism


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def baseline_reference(dataset, results, coordinator, seed, data, source):
    path = location(results, dataset, seed, coordinator, 'baseline')
    if not (path / 'final.json').exists():
        raise RuntimeError(f'Train the same-seed {NAMES[coordinator]} baseline first.')
    final = read_json(path / 'final.json')
    meta = read_json(path / 'metadata.json')
    history = [json.loads(line) for line in
               (path / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    budget = 3900 if dataset == 'office_home' else 100
    expected_source = meta.get('sources', meta.get('source_manifest', {}))
    if (len(history) != 100 or final['communication_rounds'] != budget
            or final['seed'] != seed or meta['split_sha256'] != data.digest
            or expected_source.get('sha256') != source['sha256']
            or final['identity_sha256'] != meta['identity_sha256']):
        raise RuntimeError('The baseline does not match this seed, data split, source or budget.')
    fixed = {'validation': history[-1]['validation'], 'test': final['final_test']}
    record = fixed if dataset == 'office_home' else {
        'validation': final['selected_validation'], 'test': final['selected_test']}
    if any(float(row['accuracy']) <= 0 for row in record['validation']['clients']):
        raise RuntimeError('Weight refinement requires positive baseline validation accuracies.')
    return {**record, 'fixed_epoch100': fixed, 'seed': seed, 'method': coordinator}


def _train_image_domains(dataset, data, config, entry, coordinator, mechanism,
                         device, results, rr, ww, manifest):
    is_office = dataset == 'office_home'
    domain = 'fedoffice' if is_office else 'fedretina'
    job = f'{coordinator}_{mechanism}'
    family = entry['engine']
    if mechanism == 'baseline':
        if is_office:
            return module('fedofficeseeds.runtime').run_method(
                config, data, coordinator, device, rr, ww, manifest)
        family = ('standard' if coordinator in ('fedavg', 'fedfv', 'fedawa')
                  else 'complete' if coordinator == 'fedmgda' else 'matrix')
    if family == 'refinement':
        ref = baseline_reference(dataset, results, coordinator, config['seed'], data, manifest)
        observer = module(('office' if is_office else 'retina') + '_refinement.runtime')
        return observer.run_method(config, data, job, device, rr, ww, manifest, ref)
    engine_module = domain if family == 'standard' else domain + family
    builder = module(engine_module + '.engine').build_engine
    if is_office:
        return module('fedofficecomplete.runtime').run_method(
            config, data, job, device, rr, ww, manifest, engine_factory=builder)
    method = coordinator if mechanism == 'baseline' and family == 'standard' else job
    return module('fedretina.runtime').run_method(config, data, method, device, rr, ww, manifest,
        engine_factory=builder, display_label=NAMES[coordinator] + ' / ' + mechanism)


def train(dataset, data_root, results, weights, coordinator, mechanism, seed, device):
    selected = recipe(dataset, coordinator, mechanism, seed)
    config = selected['config']
    rr = location(results, dataset, seed, coordinator, mechanism)
    ww = location(weights, dataset, seed, coordinator, mechanism)
    if rr.is_relative_to(ww) or ww.is_relative_to(rr):
        raise ValueError('Use separate results and checkpoint directories.')
    data = load_data(dataset, data_root, config)
    try:
        config['client_fit_sizes'] = list(data.fit_sizes)
        if dataset == 'retina' and 'local_steps' in config:
            config['local_steps'] = max(math.ceil(n/config['batch_size']) for n in data.fit_sizes)
        manifest = source_manifest()
        if dataset in ('fashion_mnist', 'cifar100'):
            suite = 'fashion_recipes' if dataset == 'fashion_mnist' else 'cifar_recipes'
            # These runtimes place the method below the coordinator's directory.
            final = module(suite + '.runtime').run_method(config, data, coordinator,
                selected['engine'], device, rr.parent, ww.parent, manifest)
        else:
            final = _train_image_domains(dataset, data, config, selected, coordinator,
                                         mechanism, device, results, rr, ww, manifest)
        return endpoint(final, selected['endpoint'])
    finally:
        close = getattr(data, 'close', None)
        if close:
            close()


def endpoint(result, key):
    if key == 'weight_refinement':
        selected = result.get('weight_refinement')
        if selected is None:
            raise RuntimeError('No eligible refined checkpoint was produced. Raw endpoints are in final.json.')
        return selected['test']
    if key == 'fixed_epoch':
        return result['final_test']
    if key == 'validation_selected':
        return result['selected_test']
    raise ValueError(f'Unknown endpoint: {key}')
