"""Evaluate final and averaged Office-Home checkpoints after training."""
import json
import signal
import time
from pathlib import Path
from fl_coordination._impl.fedmini.data import fingerprint
from fl_coordination._impl.fedmini.runtime import seed_all, run_lock
from fl_coordination._impl.fedpairs.common import json_write, save, load, moved, rng, set_rng, preserve_rng
from fl_coordination._impl.fedseries.runtime import bind_experiment
from fl_coordination._impl.fedoffice.data import plan_summary, DOMAINS
from fl_coordination._impl.fedoffice.runtime import one_epoch, write_curves
from .config import JOBS, settings as default_settings
from .engine import build_engine
from .observer import Observer, retention, rank


class ObservedEngine:
    def __init__(self, engine, observer):
        self.engine, self.observer = engine, observer

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def step(self, batches):
        result = self.engine.step(batches)
        self.observer.on_round(self.engine)
        return result


def run_method(cfg, data, job, device, result_dir, weight_dir, manifest, reference,
               *, engine_factory=build_engine, stop_after_epoch=None, detector_settings=None):
    rr, wr = Path(result_dir), Path(weight_dir)
    if wr.resolve().is_relative_to(rr.resolve()) or rr.resolve().is_relative_to(wr.resolve()):
        raise ValueError('Results and weights must have disjoint roots')
    rr.mkdir(parents=True, exist_ok=True); wr.mkdir(parents=True, exist_ok=True)
    grid_settings = detector_settings if detector_settings is not None else default_settings()
    experiment = bind_experiment(cfg, data, manifest, rr)
    identity = fingerprint(dict(experiment=experiment['sha256'], method=job,
                                reference=reference, detector_settings=grid_settings))
    budget = plan_summary(data, cfg)
    with run_lock(wr/'.run.lock'):
        if (rr/'final.json').exists():
            final = json.loads((rr/'final.json').read_text(encoding='utf-8'))
            if final['identity_sha256'] != identity or not all(Path(p).exists() for p in final['checkpoints'].values()):
                raise RuntimeError('Completed result identity/weights differ')
            return final
        seed_all(cfg['seed'], cfg['deterministic'])
        engine = engine_factory(cfg, job, device)
        saved = load(wr/'last.pt') if (wr/'last.pt').exists() else None
        if saved is None and (rr/'metrics.jsonl').exists():
            raise RuntimeError('Orphan metrics: refusing overwrite')
        if saved is not None and saved['identity_sha256'] != identity:
            raise RuntimeError('Resume config/source/reference changed')
        observer = Observer(engine, data, reference['validation'], grid_settings,
                            initial_losses=saved['observer']['initial_losses'] if saved else None)
        start, history, raw_best, elapsed = 0, [], None, 0.
        if saved is not None:
            engine.load_state_dict(saved['engine'])
            observer.load_state_dict(saved['observer'])
            start, history, raw_best, elapsed = (saved[k] for k in ('epoch', 'history', 'raw_best', 'elapsed_seconds'))
            set_rng(saved['rng'])
            if (len(history) != start or start > cfg['epochs'] or
                    engine.step_index != start*budget['rounds'] or observer.grid.last_step != engine.step_index):
                raise RuntimeError('Invalid resumed epoch/round/observer boundary')
            del saved
        metadata = dict(method=job, combination=JOBS[job][0], rho=cfg['rho'], config=cfg,
                        identity_sha256=identity, sources=manifest, split_sha256=data.digest,
                        initial_model_sha256=engine.initial_hash, reference=reference,
                        domains=list(DOMAINS), epoch_budget=budget, weights=str(wr.resolve()),
                        endpoints=['raw_fixed_epoch', 'weight_refinement_retention'], raw_epoch=cfg['epochs'],
                        detector_settings=grid_settings, sampling='after_completed_communication_round',
                        selection='validation_only_baseline_retention_lexicographic',
                        reporting='Raw fixed Epoch100; weight refinement uses only Epochs1-100; no feedback to training')
        json_write(rr/'metadata.json', metadata)
        data.save_manifest(rr/'split_manifest.json')
        write_curves(rr, history)
        requested = []
        def stop(signum, frame):
            requested.append(signum)
            print('Stop requested; save training and weight refinement together at Epoch boundary.', flush=True)
        handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            for epoch in range(start, cfg['epochs']):
                begin = time.monotonic(); observer.begin_epoch()
                summary, planned = one_epoch(ObservedEngine(engine, observer), data, cfg, epoch)
                validation = engine.evaluate(data, 'validation')
                record = dict(id=f'epoch_{epoch+1:03d}', epoch=epoch+1, validation=validation,
                              scores=retention(validation, reference['validation']))
                if raw_best is None or rank(record) < rank(raw_best):
                    raw_best = dict(record, model=moved(engine.model_state()))
                dense_validation = observer.end_epoch(epoch+1)
                elapsed += time.monotonic()-begin
                history.append(dict(epoch=epoch+1, round=engine.step_index, validation=validation,
                                    dense_validation=dense_validation, elapsed_seconds=elapsed,
                                    schedule_sha256=planned['sha256'], **summary))
                save_begin = time.monotonic()
                save(wr/'last.pt', dict(identity_sha256=identity, epoch=epoch+1,
                    engine=moved(engine.state_dict()), observer=observer.state_dict(), history=history,
                    raw_best=raw_best, elapsed_seconds=elapsed, rng=rng()))
                checkpoint_seconds = time.monotonic()-save_begin
                write_curves(rr, history)
                json_write(rr/'observer_audit.json', observer.audit())
                json_write(rr/'progress.json', dict(epoch=epoch+1, epochs=cfg['epochs'], round=engine.step_index,
                    last_epoch_compute_seconds=time.monotonic()-begin-checkpoint_seconds,
                    checkpoint_seconds=checkpoint_seconds, elapsed_seconds=elapsed,
                    candidates=observer.timing['candidates']))
                print(f'{job} Epoch {epoch+1}/{cfg["epochs"]} | raw val {validation["mean"]:.2f} '
                      f'worst {validation["worst"]:.2f} | valleys {observer.timing["candidates"]}', flush=True)
                if requested:
                    raise SystemExit(130)
                if stop_after_epoch is not None and epoch+1 >= stop_after_epoch:
                    return None
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        if len(history) != cfg['epochs'] or engine.step_index != cfg['epochs']*budget['rounds']:
            raise RuntimeError('Incorrect complete-data budget')
        if sum(r['samples'] for r in history) != cfg['epochs']*data.samples_per_epoch:
            raise RuntimeError('Incorrect training sample count')
        chosen = observer.finalize()
        checkpoints = dict(resume=str(wr/'last.pt'), raw=str(wr/f'raw_epoch_{cfg["epochs"]:03d}.pt'),
                           raw_validation_selected=str(wr/'raw_retention.pt'))
        save(checkpoints['raw'], dict(identity_sha256=identity, epoch=cfg['epochs'],
                                     model=moved(engine.model_state()), validation=history[-1]['validation']))
        save(checkpoints['raw_validation_selected'], dict(identity_sha256=identity, **raw_best))
        selection = dict(identity_sha256=identity, rule='baseline_retention', reference=reference,
                         budget_epoch=cfg['epochs'], raw_validation_selected={k:v for k,v in raw_best.items() if k!='model'},
                         weight_refinement={k:v for k,v in chosen.items() if k!='state'} if chosen else None,
                         weight_refinement_status='selected' if chosen else 'no_eligible_valley')
        if chosen:
            checkpoints['weight_refinement'] = str(wr/'weight_refinement_retention.pt')
            save(checkpoints['weight_refinement'], dict(identity_sha256=identity, model=chosen['state'], **selection['weight_refinement']))
        # Freeze the validation decision before any test access.
        json_write(rr/'selection.json', selection)
        json_write(rr/'observer_audit.json', observer.audit())
        with preserve_rng():
            raw_test = engine.evaluate(data, 'test')
            observer.scratch.load_model(raw_best['model'])
            raw_diagnostic = observer.scratch.evaluate(data, 'test')
            refinement_test = None
            if chosen:
                observer.scratch.load_model(chosen['state'])
                refinement_test = observer.scratch.evaluate(data, 'test')
        base_mean = reference['test']['mean']
        def endpoint(test, val):
            return dict(test=test, validation=val, improvement_pp=test['mean']-base_mean)
        final = dict(identity_sha256=identity, method=job, combination=JOBS[job][0], rho=cfg['rho'],
            epochs=cfg['epochs'], seed=cfg['seed'], communication_rounds=engine.step_index,
            training_samples=sum(r['samples'] for r in history), elapsed_seconds=elapsed,
            raw=endpoint(raw_test, history[-1]['validation']),
            weight_refinement=endpoint(refinement_test, chosen['validation']) if chosen else None,
            weight_refinement_status=selection['weight_refinement_status'],
            raw_validation_selected=dict(epoch=raw_best['epoch'], **endpoint(raw_diagnostic, raw_best['validation'])),
            checkpoints=checkpoints, split_sha256=data.digest, initial_model_sha256=engine.initial_hash,
            final_test=raw_test, reporting_epoch=cfg['epochs'])
        json_write(rr/'final.json', final)
        # Once all selected models and final metrics exist, release unselected valley tensors
        # from the recovery file. Keep raw optimizer/coordinator/RNG state and scalar audits.
        # The complete-result guard runs before resume loading, including after a crash here.
        save(wr/'last.pt', dict(identity_sha256=identity, epoch=cfg['epochs'],
             engine=moved(engine.state_dict()), history=history, rng=rng(),
             elapsed_seconds=elapsed, completed=True, selection=selection,
             observer_summary=observer.audit()))
        print(f'DONE {job}: raw {raw_test["mean"]:.2f}; weight refinement '+
              (f'{refinement_test["mean"]:.2f}' if refinement_test else 'no eligible valley'), flush=True)
        return final
