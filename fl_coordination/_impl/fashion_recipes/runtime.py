"""Dispatch the configured Fashion-MNIST training recipe."""
from functools import partial
from pathlib import Path
from types import FunctionType
from fl_coordination._impl.fedseries import runtime as frozen
from fl_coordination._impl.fedteam.common import write
from fl_coordination._impl.fedavg.engine import Engine as AverageEngine
from fl_coordination._impl.feda3.engine import Engine as ExtraEngine
from fl_coordination._impl.fedseries.engine import Engine as SeriesEngine
from .protocol import label, display


def engine_factory(config, coordinator, variant, device, network=None):
    if variant == 'fedgloss':
        from .local_gloss import LocalizedGloSSEngine
        return LocalizedGloSSEngine(config, coordinator, variant, device, network=network)
    if variant == 'fedgf':
        from fl_coordination._impl.fedglobalflat.engine import Engine
        return Engine(config, coordinator, variant, device, network=network)
    if coordinator == 'fedavg':
        engine = AverageEngine
    elif coordinator in ('fedfv', 'fedmgda'):
        engine = SeriesEngine
    else:
        engine = ExtraEngine
    return engine(config, coordinator, variant, device, network=network)


def write_metadata(path, value, *, local_probe=None):
    if path.name in ('metadata.json', 'final.json'):
        identity = value.get('identity', value)
        c, v = identity['coordinator'], identity['variant']
        value = dict(value, display_name=display(c,v),
            perturbation={'baseline':'none', 'split':'ASAM', 'sam':'ASAM',
                          'fedgf':'FedGF', 'fedgloss':'FedGloSS'}[v],
            branch_policy='ordinary' if v=='baseline' else 'coordinate(g+h)' if v=='sam' else 'coordinate(g)+coordinate(h)',
            perturbation_scope='none' if v=='baseline' else 'shared_encoder_only' if v in ('fedgf','fedgloss') else 'shared_encoder_and_private_head')
        if local_probe is not None:
            value.update(
                perturbation='FedGloSS+LocalSAM',
                rho_global=local_probe['rho'],
                rho_local=local_probe['local_sam_rho'],
                probe_policy=local_probe['probe_policy'],
            )
    write(path, value)


def run_method(config, data, coordinator, variant, device, result_root, weight_root,
               source_manifest, **kwargs):
    name = label(coordinator, variant)
    source = frozen.run_method
    if variant in ('fedgf', 'fedgloss'):
        from fl_coordination._impl.fedglobalflat.runtime import run_method as source
    # Different perturbations have different config identities. Bind inside the
    # exact method directory, avoiding a conflict in its coordinator parent.
    def bind(cfg, dataset, manifest, root):
        return frozen.bind_experiment(cfg, dataset, manifest, Path(root)/name)
    bindings = dict(source.__globals__, method_label=label, json_write=write_metadata,
                    bind_experiment=bind)
    if variant == 'fedgloss':
        from .local_gloss import epoch_summary
        bindings.update(epoch_summary=epoch_summary,
                        json_write=partial(write_metadata, local_probe=config))
    runner = FunctionType(source.__code__, bindings, name='run_method', argdefs=source.__defaults__)
    runner.__kwdefaults__ = dict(source.__kwdefaults__, engine_factory=engine_factory)
    return runner(config, data, coordinator, variant, device, result_root, weight_root, source_manifest, **kwargs)
