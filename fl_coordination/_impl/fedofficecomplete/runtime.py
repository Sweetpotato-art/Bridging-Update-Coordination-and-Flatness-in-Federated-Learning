"""Complete exactly 100 data Epochs and report the fixed final checkpoint."""
import json
import signal
import time
from pathlib import Path
from fl_coordination._impl.fedmini.data import fingerprint
from fl_coordination._impl.fedmini.runtime import seed_all, run_lock
from fl_coordination._impl.fedpairs.common import json_write, save, load, moved, rng, set_rng
from fl_coordination._impl.fedseries.runtime import bind_experiment
from fl_coordination._impl.fedoffice.data import plan_summary, DOMAINS
from fl_coordination._impl.fedoffice.runtime import one_epoch, permanent_checkpoints, write_curves
from .config import label, parse_job
from .engine import build_engine


def run_method(cfg, data, job, device, result_dir, weight_dir, manifest,
               *, engine_factory=build_engine, stop_after_epoch=None):
    rr, wr = Path(result_dir), Path(weight_dir)
    if wr.resolve().is_relative_to(rr.resolve()) or rr.resolve().is_relative_to(wr.resolve()):
        raise ValueError('Results and weights require separate roots')
    rr.mkdir(parents=True, exist_ok=True)
    wr.mkdir(parents=True, exist_ok=True)
    experiment = bind_experiment(cfg, data, manifest, rr)
    identity = fingerprint(dict(experiment=experiment['sha256'], method=job))
    budget = plan_summary(data, cfg)
    last_path, best_path = wr/'last.pt', wr/'best.pt'
    milestones = cfg['checkpoint_epochs']
    with run_lock(wr/'.run.lock'):
        if (rr/'final.json').exists():
            final = json.loads((rr/'final.json').read_text(encoding='utf-8'))
            if final['identity_sha256'] != identity:
                raise RuntimeError('Completed experiment identity differs')
            required = [last_path, best_path, *[wr/f'epoch_{e:03d}.pt' for e in milestones],
                        *[wr/f'best_until_{e:03d}.pt' for e in milestones]]
            if not all(p.exists() for p in required):
                raise RuntimeError('Completed experiment has missing weights')
            print('Already complete: '+job, flush=True)
            return final
        seed_all(cfg['seed'], cfg['deterministic'])
        engine = engine_factory(cfg, job, device)
        start, history, best, elapsed = 0, [], None, 0.
        if last_path.exists():
            state = load(last_path)
            if state['identity_sha256'] != identity:
                raise RuntimeError('Resume source/config/data identity changed')
            engine.load_state_dict(state['engine'])
            set_rng(state['rng'])
            start, history, best, elapsed = (state[k] for k in ('epoch', 'history', 'best', 'elapsed_seconds'))
            if start>cfg['epochs']:raise RuntimeError('Checkpoint exceeds the fixed budget')
            if len(history) != start or engine.step_index != start*budget['rounds']:
                raise RuntimeError('Resume history/round mismatch')
            selected = load(best_path) if best_path.exists() else None
            if selected is None or selected['identity_sha256'] != identity or selected['epoch'] != best['epoch']:
                if best['epoch'] != start:
                    raise RuntimeError('Missing earlier selected weights')
                save(best_path, dict(identity_sha256=identity, epoch=start, validation=best['validation'],
                                     model=moved(engine.model_state())))
            del selected
            permanent_checkpoints(state, wr, milestones)
            for e in milestones:
                if e <= start and (not (wr/f'epoch_{e:03d}.pt').exists() or
                                   not (wr/f'best_until_{e:03d}.pt').exists()):
                    raise RuntimeError('Missing earlier permanent checkpoint')
            del state
        elif (rr/'metrics.jsonl').exists() or best_path.exists():
            raise RuntimeError('Orphan outputs; refusing overwrite')
        c, v = parse_job(job)
        metadata = dict(method=job, display_method=label(job), identity_sha256=identity,
            config=cfg, sources=manifest, split_sha256=data.digest,
            initial_model_sha256=engine.initial_hash, epoch_budget=budget,
            total_rounds=budget['rounds']*cfg['epochs'], total_samples=data.samples_per_epoch*cfg['epochs'],
            weights=str(wr.resolve()), domains=list(DOMAINS),
            selection='Highest arithmetic mean validation accuracy; earliest tie',
            evaluation='Held-out samples from participating domains',
            perturbation={'baseline':'none','sam':'Standard SAM, joint shared/private',
                'asam':'Adaptive SAM, joint shared/private','gf':'FedGF, shared coordinates',
                'gloss':'Historical global shift plus per-batch local SAM, shared coordinates'}[v],
            coordination='A(mean g)' if v=='baseline' else ('A(mean p)' if v=='asam' else 'A_g(mean g) + A_h(mean(p-g))'),
            coordinator=c,independent_branch_state=v in ('sam','gf','gloss'),
            primary_reporting='Fixed final Epoch checkpoint test; validation-best is diagnostic only')
        json_write(rr/'metadata.json', metadata)
        data.save_manifest(rr/'split_manifest.json')
        write_curves(rr, history)
        requested = []
        def stop(signum, frame):
            requested.append(signum)
            print('Stop requested: saving at the next complete Epoch boundary.', flush=True)
        handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for epoch in range(start, cfg['epochs']):
                begin = time.monotonic()
                summary, planned = one_epoch(engine, data, cfg, epoch)
                validation = engine.evaluate(data, 'validation')
                elapsed += time.monotonic()-begin
                history.append(dict(epoch=epoch+1, round=engine.step_index, validation=validation,
                    elapsed_seconds=elapsed, schedule_sha256=planned['sha256'], **summary))
                improved = best is None or validation['mean'] > best['validation']['mean']
                if improved:
                    best = dict(epoch=epoch+1, validation=validation)
                state = dict(identity_sha256=identity, epoch=epoch+1, engine=moved(engine.state_dict()),
                             history=history, best=best, rng=rng(), elapsed_seconds=elapsed)
                save(last_path, state)
                if improved:
                    save(best_path, dict(identity_sha256=identity, epoch=epoch+1, validation=validation,
                                         model=moved(engine.model_state())))
                permanent_checkpoints(state, wr, milestones)
                del state
                write_curves(rr, history)
                print(f'{label(job)} | Epoch {epoch+1}/{cfg["epochs"]} | validation '
                      f'{validation["mean"]:.2f}% | selected {best["epoch"]}', flush=True)
                if epoch+1 in milestones:
                    json_write(rr/f'epoch_{epoch+1:03d}.json', dict(epoch=epoch+1, validation=validation,
                        best_so_far=best, checkpoint=str(wr/f'epoch_{epoch+1:03d}.pt')))
                if requested:
                    raise SystemExit(130)
                if stop_after_epoch is not None and epoch+1 >= stop_after_epoch:
                    return None
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        if start > cfg['epochs'] or len(history)!=cfg['epochs']:
            raise RuntimeError('Training exceeds or misses the fixed budget')
        fixed=load(wr/f"epoch_{cfg['epochs']:03d}.pt")
        if fixed['epoch']!=cfg['epochs'] or fixed['identity_sha256']!=identity:
            raise RuntimeError('Incorrect fixed-epoch checkpoint')
        engine.load_state_dict(fixed['engine']);del fixed
        final_test = engine.evaluate(data, 'test')
        selected = load(best_path)
        if selected['identity_sha256'] != identity:
            raise RuntimeError('Selected checkpoint identity differs')
        engine.load_model(selected['model'])
        selected_test = engine.evaluate(data, 'test')
        final = dict(identity_sha256=identity, method=job, display_method=label(job), seed=cfg['seed'],
            epochs=cfg['epochs'], selected_epoch=best['epoch'], selected_validation=best['validation'],
            selected_test=selected_test, final_test=final_test, primary_test=final_test,
            reporting_epoch=cfg['epochs'], reporting='fixed_epoch_checkpoint', initial_model_sha256=engine.initial_hash,
            elapsed_seconds=elapsed, communication_rounds=engine.step_index,
            training_samples=sum(r['samples'] for r in history), checkpoint_best=str(best_path),
            checkpoint_resume=str(last_path), permanent_checkpoints={str(e):str(wr/f'epoch_{e:03d}.pt') for e in milestones})
        if final['training_samples'] != metadata['total_samples'] or final['communication_rounds'] != metadata['total_rounds']:
            raise RuntimeError('Incorrect full-data training budget')
        json_write(rr/'final.json', final)
        print(f'DONE {label(job)}: fixed test {final_test["mean"]:.2f}%, worst {final_test["worst"]:.2f}%', flush=True)
        return final
