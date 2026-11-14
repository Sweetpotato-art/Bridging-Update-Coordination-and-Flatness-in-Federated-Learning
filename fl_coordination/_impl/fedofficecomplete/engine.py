"""Office-Home coordination adapters for baseline and flatness recipes."""
from fl_coordination._impl.fedcifar.engine import Engine as TrajectoryEngine
from fl_coordination._impl.fedcifarglobalflat.engine import Engine as FlatEngine
from fl_coordination._impl.fedcifarextra.operators import make_coordinator
from fl_coordination._impl.fedglobalflat.mechanisms import Mechanism
from fl_coordination._impl.local_probe.engine import Engine as LocalProbeEngine
from fl_coordination._impl.fedoffice.model import build_network
from .config import parse_job


def initialize_flat(engine,cfg,c,kind,device,network):
    TrajectoryEngine.__init__(engine,cfg,c,'split',device,network)
    if c=='fedawa':
        layout=[(n,p.numel()) for n,p in engine.encoder.named_parameters()]
        engine.g_coordinator=make_coordinator(c,cfg,device,layout)
        engine.h_coordinator=make_coordinator(c,cfg,device,layout)
    counts=cfg['client_fit_sizes']
    if len(counts)!=cfg['clients'] or any(type(n)!=int or n<=0 for n in counts):
        raise ValueError('Exact fitting sample counts required')
    engine.fractions=[n/sum(counts) for n in counts]
    engine.mechanism=Mechanism(kind,cfg,device)


class GFEngine(FlatEngine):
    def __init__(self,cfg,c,device,network):initialize_flat(self,cfg,c,'fedgf',device,network)
    step=FlatEngine.step
    state_dict=FlatEngine.state_dict
    load_state_dict=FlatEngine.load_state_dict


class GloSSLocalEngine(LocalProbeEngine):
    def __init__(self,cfg,c,device,network):
        initialize_flat(self,cfg,c,'fedgloss',device,network)
        self.mechanism.local_sam_rho=cfg['local_sam_rho']
        self.mechanism.local_probe_records=[]


def build_engine(cfg,job,device,network=None):
    c,v=parse_job(job)
    if cfg['coordinator']!=c or cfg['mechanism']!=v or cfg['head_mode']!='private' or cfg['adaptive_sam']!=(v=='asam'):
        raise ValueError('Configuration/engine mismatch')
    net=network if network is not None else build_network(cfg)
    if v=='gloss':engine=GloSSLocalEngine(cfg,c,device,net)
    elif v=='gf':engine=GFEngine(cfg,c,device,net)
    else:engine=TrajectoryEngine(cfg,c,{'baseline':'baseline','sam':'split','asam':'sam'}[v],device,net)
    for group in engine.shared_optimizer.param_groups:group['lr']=cfg['server_learning_rate']
    return engine
