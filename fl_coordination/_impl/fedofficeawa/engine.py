"""Office-Home training with FedAWA-Agg."""
from fl_coordination._impl.fedcifarextra.engine import Engine as UnequalStepEngine
from fl_coordination._impl.fedoffice.model import build_network


class Engine(UnequalStepEngine):
    def __init__(self,config,device,network=None):
        if config['rho']!=0 or config['flat_lambda']!=0 or config['adaptive_sam']:
            raise ValueError('Expected an unperturbed FedAWA-Agg recipe')
        super().__init__(config,'fedawa','baseline',device,network or build_network(config))
        for group in self.shared_optimizer.param_groups:
            group['lr']=config['server_learning_rate']


def build_engine(config,method,device,network=None):
    if method!='fedawa':
        raise ValueError('Expected the Office-Home FedAWA-Agg baseline')
    return Engine(config,device,network)
