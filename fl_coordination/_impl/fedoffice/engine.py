"""Office-Home training with FedAvg, FedFV, and directly coordinated ASAM."""
from fl_coordination._impl.fedseries.engine import Engine as TrajectoryEngine
from fl_coordination._impl.fedcifaravg.engine import Engine as FedAvgEngine
from .model import build_network

METHODS=('fedavg','fedfv','fedfv_asam')
LABELS={'fedavg':'FedAvg','fedfv':'FedFV','fedfv_asam':'FedFV & FedASAM'}


class FVEngine(TrajectoryEngine):
    def __init__(self, config, method, device, network=None):
        variant='sam' if method=='fedfv_asam' else 'baseline'
        super().__init__(config,'fedfv',variant,device,network or build_network(config))
        for group in self.shared_optimizer.param_groups:
            group['lr']=config['server_learning_rate']


def build_engine(config, method, device, network=None):
    if method not in METHODS:
        raise ValueError('Unsupported Office-Home training recipe')
    if method=='fedavg':
        return FedAvgEngine(config,'fedavg','baseline',device,network or build_network(config))
    return FVEngine(config,method,device,network)


def method_config(common,method,data):
    cfg=dict(common,client_fit_sizes=list(data.fit_sizes))
    if method=='fedavg':
        cfg.update(optimizer='none',rho=0.,flat_lambda=0.)
    elif method=='fedfv':
        cfg.update(rho=0.,flat_lambda=0.)
    elif method!='fedfv_asam':
        raise ValueError(method)
    return cfg
