from fl_coordination.recipes import NAMES

VARIANTS = dict(baseline='baseline', split='sam', sam='asam',
                fedgf='gf', fedgloss='gloss')
def label(coordinator, variant):
    return VARIANTS[variant]
def display(coordinator, variant):
    return NAMES[coordinator] + {'baseline':'', 'split':' + ASAM (separate)',
        'sam':' + ASAM (direct)', 'fedgf':' + FedGF', 'fedgloss':' + FedGloSS'}[variant]
