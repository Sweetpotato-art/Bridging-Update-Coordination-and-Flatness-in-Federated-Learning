from fl_coordination.recipes import NAMES

def parse_job(job):
    coordinator, mechanism = job.rsplit('_', 1)
    return coordinator, mechanism

def label(job):
    coordinator, mechanism = parse_job(job)
    return NAMES[coordinator] + LABEL_SUFFIX[mechanism]

LABEL_SUFFIX = {'baseline':'', 'sam':' + FedSAM', 'asam':' + FedASAM',
                'gf':' + FedGF', 'gloss':' + FedGloSS with local SAM'}
