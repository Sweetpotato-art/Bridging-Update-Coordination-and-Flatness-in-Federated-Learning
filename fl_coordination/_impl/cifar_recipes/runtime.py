"""Dispatch the configured CIFAR-100 training recipe."""
from types import FunctionType
from pathlib import Path
from .protocol import label, display, variant


def implementation(c,m):
    if c=='fedavg':
        from fl_coordination._impl.fedcifaravg.runtime import run_method
    elif m=='fedgloss_local':
        if c=='fedfv':
            from fl_coordination._impl.local_probe.runtime import run_method
        else:
            from fl_coordination._impl.localized_gloss.runtime import run_method
    elif c=='fedmgda' and m in ('fedasam','fedgf'):
        from fl_coordination._impl.cifar_mgda.runtime import gf_run, sam_run
        run_method = gf_run if m == 'fedgf' else sam_run
    elif m=='fedgf':
        from fl_coordination._impl.fedcifarglobalflat.runtime import run_method
    elif m=='fedasam':
        from fl_coordination._impl.fedcifarasam.runtime import run_method
    elif c in ('fedmdfg','fedawa') or c=='fedmgda' and m=='baseline':
        from fl_coordination._impl.fedcifarextra.runtime import run_method
    else:
        from fl_coordination._impl.fedcifar.runtime import run_method
    return run_method


def run_method(config,data,c,m,device,rr,wr,manifest,**kwargs):
    from fl_coordination._impl.fedseries.runtime import bind_experiment
    source=implementation(c,m)
    name=label(c,m)
    def bind(cfg,dataset,src,root):
        return bind_experiment(cfg,dataset,src,Path(root)/name)
    runner=FunctionType(source.__code__,dict(source.__globals__,
        method_label=lambda *a:name,display_label=lambda *a:display(c,m),bind_experiment=bind),
        argdefs=source.__defaults__)
    runner.__kwdefaults__=dict(source.__kwdefaults__)
    return runner(config,data,c,variant(m),device,rr,wr,manifest,**kwargs)
