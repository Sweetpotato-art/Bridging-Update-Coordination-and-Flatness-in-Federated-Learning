def settings():
    return [
        dict(id=f'w{warmup}_z{str(ratio).replace(".", "p")}', warmup_epochs=warmup,
             tolerance_ratio=ratio, n_converge=3, n_tolerance=6,
             descent_confirmations=3, minimum_valley_segments=3)
        for warmup, ratio in ((20, .3), (50, .1), (50, .3), (50, .5), (80, .3))
    ]


class Combinations:
    def __getitem__(self, job):
        return job, None

JOBS = Combinations()
