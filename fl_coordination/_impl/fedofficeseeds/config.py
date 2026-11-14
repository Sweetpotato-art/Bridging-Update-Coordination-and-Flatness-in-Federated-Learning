from fl_coordination.recipes import NAMES

METHODS = tuple(NAMES)

def build_engine(cfg, method, device, network=None):
    if method in ('fedavg', 'fedfv'):
        from fl_coordination._impl.fedoffice.engine import build_engine as factory
        return factory(cfg, method, device, network)
    if method == 'fedawa':
        from fl_coordination._impl.fedofficeawa.engine import build_engine as factory
        return factory(cfg, method, device, network)
    if method == 'fedmgda':
        from fl_coordination._impl.fedofficecomplete.engine import build_engine as factory
    else:
        from fl_coordination._impl.fedofficematrix.engine import build_engine as factory
    return factory(cfg, method+'_baseline', device, network)
