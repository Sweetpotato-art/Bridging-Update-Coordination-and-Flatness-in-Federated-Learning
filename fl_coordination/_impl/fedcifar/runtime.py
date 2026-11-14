"""Complete CIFAR-100 runs, held-out selection, exact resume, separate weights."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from fl_coordination._impl.fedmini.data import fingerprint
from fl_coordination._impl.fedmini.runtime import seed_all, run_lock, write_history
from fl_coordination._impl.fedpairs.common import json_write, load, moved, rng, save, set_rng
from fl_coordination._impl.fedseries.runtime import bind_experiment, epoch_summary
from .data import epoch_plan, plan_summary
from .engine import Engine, method_label, display_label


def run_method(config, data, coordinator, variant, device, result_root, weight_root,
               source_manifest, *, engine_factory=Engine, stop_after_epoch=None):
    """Train CIFAR-100 and evaluate the validation-selected checkpoint."""
    label = method_label(coordinator, variant, config)
    result_dir, weight_dir = Path(result_root) / label, Path(weight_root) / label
    result_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    experiment = bind_experiment(config, data, source_manifest, result_root)
    identity = dict(experiment=experiment['sha256'], coordinator=coordinator, variant=variant)
    identity_hash = fingerprint(identity)
    budget = plan_summary(data, config)
    last_path, best_path = weight_dir / 'last.pt', weight_dir / 'best.pt'
    final_path, meta_path = result_dir / 'final.json', result_dir / 'metadata.json'
    with run_lock(weight_dir / '.run.lock'):
        metadata = dict(identity=identity, identity_sha256=identity_hash, label=label,
                        protocol='CIFAR-100 shared-backbone training with private classifiers',
                        head_mode=config['head_mode'], bn_policy=config['bn_policy'],
                        config=config, sources=source_manifest, split_sha256=data.digest,
                        weights=str(weight_dir.resolve()), epoch_budget=budget,
                        total_rounds=budget['rounds'] * config['epochs'],
                        total_samples=data.samples_per_epoch * config['epochs'],
                        rho=config['rho'] if variant != 'baseline' else 0.,
                        display_method=display_label(coordinator, variant, config),
                        perturbation=('none' if variant == 'baseline' else
                                      'ASAM' if config['adaptive_sam'] else 'SAM'),
                        coordination={'baseline': 'ordinary', 'sam': 'direct',
                                      'split': 'independent_g_h'}[variant],
                        selection='Maximum all-client mean validation accuracy; earliest tie',
                        communication='Theoretical dense FP32 shared vectors; no compression or network timing',
                        task='identical CIFAR-100 labels 0..99 at every client')
        if meta_path.exists():
            old = json.loads(meta_path.read_text(encoding='utf-8'))
            if old['identity_sha256'] != identity_hash:
                raise RuntimeError('Run settings changed. Use new --results and --weights roots.')
            if 'initial_model_sha256' in old:
                metadata['initial_model_sha256'] = old['initial_model_sha256']
        if final_path.exists():
            final = json.loads(final_path.read_text(encoding='utf-8'))
            if final['identity_sha256'] != identity_hash:
                raise RuntimeError('Completed method identity differs')
            if not last_path.exists() or not best_path.exists():
                raise RuntimeError('Completed logs exist but weights are missing; restore weights')
            print(f'{config["case"]} {label}: complete; preserving existing result.', flush=True)
            return final

        seed_all(config['seed'], config['deterministic'])
        engine = engine_factory(config, coordinator, variant, device)
        start, history, best, elapsed = 0, [], None, 0.
        if last_path.exists():
            state = load(last_path)
            if state['identity_sha256'] != identity_hash:
                raise RuntimeError('Resume identity differs from current source/config/data')
            engine.load_state_dict(state['engine'])
            start, history, best, elapsed = state['epoch'], state['history'], state['best'], state['elapsed_seconds']
            set_rng(state['rng'])
            if engine.step_index != start * budget['rounds'] or len(history) != start:
                raise RuntimeError('Resume epoch/round/history counts are inconsistent')
            valid_best = False
            if best_path.exists():
                stored = load(best_path)
                valid_best = stored['identity_sha256'] == identity_hash and stored['epoch'] == best['epoch']
                del stored
            if not valid_best:
                if best['epoch'] != start:
                    raise RuntimeError('Earlier selected checkpoint is missing; restore best.pt')
                save(best_path, dict(identity_sha256=identity_hash, epoch=start,
                                     validation=best['validation'], model=moved(engine.model_state())))
            del state
            print(f'{config["case"]} {label}: resume epoch {start}/{config["epochs"]}.', flush=True)
        elif (result_dir / 'metrics.jsonl').exists() or best_path.exists():
            raise RuntimeError('Logs/best checkpoint without authoritative last.pt; refusing overwrite')
        metadata['initial_model_sha256'] = engine.initial_hash
        json_write(meta_path, metadata)
        write_history(result_dir / 'metrics.jsonl', history)
        print(f'{config["case"]} {label}: {config["epochs"]} complete data epochs, '
              f'{metadata["total_rounds"]} communication rounds, {metadata["total_samples"]:,} samples; '
              f'{config["participants"]}/{config["clients"]} clients, K<={config["local_steps"]}, rho={metadata["rho"]}.', flush=True)
        for epoch in range(start, config['epochs']):
            begin = time.monotonic()
            records = []
            plan = epoch_plan(data, config, epoch)
            planned = plan_summary(data, config, epoch)
            if len(plan) != budget['rounds']:
                raise RuntimeError('Per-epoch communication budget unexpectedly changed')
            for round_in_epoch, selected in enumerate(plan, 1):
                batches = {cid: [data.batch('fit', ids, device, epoch=epoch, cid=cid) for ids in chunks]
                           for cid, chunks in selected.items()}
                record = engine.step(batches)
                records.append(record)
                if round_in_epoch % config['log_every_rounds'] == 0 or round_in_epoch == len(plan):
                    print(f'{config["case"]} {label} | epoch {epoch+1}/{config["epochs"]} | '
                          f'round {engine.step_index}/{metadata["total_rounds"]} | '
                          f'clients {record["client_ids"]} | CE {np.mean(record["losses"]):.4f}', flush=True)
            summary = epoch_summary(records, config['clients'])
            if summary['client_samples'] != data.fit_sizes or summary['client_steps'] != data.steps_per_client:
                raise RuntimeError('Actual training coverage differs from planned epoch coverage')
            validation = engine.evaluate(data, 'validation')
            elapsed += time.monotonic() - begin
            history.append(dict(epoch=epoch+1, round=engine.step_index,
                                samples_seen=(epoch+1)*data.samples_per_epoch,
                                validation=validation, schedule_sha256=planned['sha256'],
                                elapsed_seconds=elapsed, **summary))
            improved = best is None or validation['mean'] > best['validation']['mean']
            if improved:
                best = dict(epoch=epoch+1, validation=validation)
            state = dict(identity_sha256=identity_hash, epoch=epoch+1, engine=moved(engine.state_dict()),
                         history=history, best=best, rng=rng(), elapsed_seconds=elapsed)
            save(last_path, state)
            del state
            if improved:
                save(best_path, dict(identity_sha256=identity_hash, epoch=epoch+1,
                                     validation=validation, model=moved(engine.model_state())))
            write_history(result_dir / 'metrics.jsonl', history)
            accuracies = ' | '.join(f'C{i+1} {c["accuracy"]:.2f}%' for i, c in enumerate(validation['clients']))
            print(f'{config["case"]} {label} | {accuracies} | mean {validation["mean"]:.2f}% '
                  f'| selected epoch {best["epoch"]}', flush=True)
            if stop_after_epoch is not None and epoch+1 >= stop_after_epoch:
                return None
        final_test = engine.evaluate(data, 'test')
        selected = load(best_path)
        if selected['identity_sha256'] != identity_hash:
            raise RuntimeError('Selected checkpoint identity differs')
        engine.load_model(selected['model'])
        selected_test = engine.evaluate(data, 'test')
        final = dict(identity_sha256=identity_hash, experiment_sha256=experiment['sha256'],
                     case=config['case'], method=label, coordinator=coordinator, variant=variant,
                     display_method=metadata['display_method'], perturbation=metadata['perturbation'],
                     coordination=metadata['coordination'], rho=metadata['rho'],
                     seed=config['seed'], epochs=config['epochs'], communication_rounds=engine.step_index,
                     training_samples=sum(r['samples'] for r in history),
                     local_steps=sum(sum(r['client_steps']) for r in history),
                     backward_passes=sum(r['backward_passes'] for r in history),
                     uplink_vector_bytes=sum(r['uplink_vector_bytes'] for r in history),
                     downlink_parameter_bytes=sum(r['downlink_parameter_bytes'] for r in history),
                     initial_model_sha256=engine.initial_hash, selected_epoch=best['epoch'],
                     selected_validation=best['validation'], selected_test=selected_test,
                     final_test=final_test, elapsed_seconds=elapsed,
                     checkpoint_best=str(best_path.resolve()), checkpoint_resume=str(last_path.resolve()))
        if final['training_samples'] != metadata['total_samples'] or final['communication_rounds'] != metadata['total_rounds']:
            raise RuntimeError('Final training budget did not match declared budget')
        json_write(final_path, final)
        print(f'{config["case"]} {label}: DONE. Selected test {selected_test["mean"]:.2f}%; '
              f'final test {final_test["mean"]:.2f}%.', flush=True)
        return final
