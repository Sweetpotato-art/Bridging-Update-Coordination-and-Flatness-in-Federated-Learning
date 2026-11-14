"""Existing Retina numerical kernels; no changes to the raw training trajectory."""
from fl_coordination._impl.fedretinacomplete.engine import GFEngine,GloSSLocalEngine
from fl_coordination._impl.fedretinamatrix.engine import SAMEngine
from fl_coordination._impl.fedretina.model import build_network

def build_engine(cfg,job,device,network=None):
    combination,rho=job,cfg["rho"];c,v=combination.rsplit('_',1)
    if (cfg['coordinator'],cfg['mechanism'],cfg['rho'],cfg['adaptive_sam'])!=(c,v,rho,v=='asam'):raise ValueError('Job/config mismatch')
    net=network if network is not None else build_network(cfg)
    if v=='gf':engine=GFEngine(cfg,c,device,net)
    elif v=='gloss':engine=GloSSLocalEngine(cfg,c,device,net)
    else:engine=SAMEngine(cfg,c,'sam' if v=='asam' else 'split',device,net)
    for group in engine.shared_optimizer.param_groups:group['lr']=cfg['server_learning_rate']
    return engine
