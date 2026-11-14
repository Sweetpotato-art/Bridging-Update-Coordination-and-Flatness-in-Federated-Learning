from fl_coordination._impl.fedmini.coordinators import FedMGDAPlus

def make_coordinator(name, config, device, layout):
    if name != 'fedmgda':
        raise ValueError('The minimum-norm adapter requires FedMGDA+.')
    return FedMGDAPlus(config, device)
