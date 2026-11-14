from fl_coordination.recipes import NAMES

FLAT_NAMES = {'fedgloss': 'FedGloSS'}
NEW_JOBS = {(coordinator, 'fedgloss') for coordinator in NAMES if coordinator != 'fedavg'}
