"""Keep the verified local-SAM trajectory and select the requested coordinator."""
import math
from fl_coordination._impl.cifar_probe_engine.engine import Engine as CoordinatorEngine
from fl_coordination._impl.local_probe.engine import Engine as LocalProbeEngine
from .config import NAMES


def method_label(coordinator, mechanism, config=None):
    if coordinator not in NAMES or mechanism != 'fedgloss':
        raise ValueError('Only the five new FedGloSS/local-SAM combinations are enabled')
    return NAMES[coordinator]+'-FedGloSS-LocalSAM-split'


def display_label(coordinator, mechanism, config=None):
    method_label(coordinator, mechanism, config)
    return NAMES[coordinator]+' & FedGloSS + local SAM (independent g/h)'


class Engine(LocalProbeEngine):
    def __init__(self, config, coordinator, variant, device, network=None):
        method_label(coordinator, variant, config)
        local_rho = config['local_sam_rho']
        if not math.isfinite(local_rho) or local_rho < 0:
            raise ValueError('Invalid local SAM radius')
        # The expansion initializer already selects the pinned MDFG/AWA/LD/MGDA/LF
        # operator. Inherit the exact FedFV local-SAM step and checkpoint methods.
        CoordinatorEngine.__init__(self, config, coordinator, variant, device, network)
        self.mechanism.local_sam_rho = local_rho
        self.mechanism.local_probe_records = []
