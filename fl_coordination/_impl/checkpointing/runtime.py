import shutil
from fl_coordination._impl.fedpairs.common import save, load

def milestones(state,wr,config):
    r = state['round']
    if r not in config['checkpoint_rounds']:return
    path = wr/f'round_{r:04d}.pt'
    if not path.exists():save(path,state)
    else:
        old = load(path)
        if old['identity_sha256']!=state['identity_sha256'] or old['round']!=r:
            raise RuntimeError('Conflicting permanent checkpoint')
    selected = wr/f'best_until_{r:04d}.pt'
    if not selected.exists():
        tmp=selected.with_suffix('.pt.tmp');shutil.copyfile(wr/'best.pt',tmp);tmp.replace(selected)
