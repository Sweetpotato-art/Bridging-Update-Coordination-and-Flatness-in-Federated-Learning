from fl_coordination.recipes import NAMES

VARIANTS = dict(baseline='baseline', fedsam='split', fedasam='sam', fedgf='fedgf',
                fedgloss_local='fedgloss')
LABELS = dict(baseline='', fedsam=' + FedSAM', fedasam=' + FedASAM',
              fedgf=' + FedGF', fedgloss_local=' + FedGloSS with local SAM')
def label(coordinator, method):
    return dict(baseline="baseline", fedsam="sam", fedasam="asam", fedgf="gf", fedgloss_local="gloss")[method]
def variant(method):
    return VARIANTS[method]
def display(coordinator, method):
    return NAMES[coordinator] + LABELS[method]
