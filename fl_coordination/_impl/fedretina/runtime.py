"""Each round covers every fitting sample once; validation selects all checkpoints."""
import json
import time
from pathlib import Path
from fl_coordination._impl.checkpointing.runtime import milestones
from fl_coordination._impl.fedmini.data import fingerprint
from fl_coordination._impl.fedmini.runtime import seed_all, run_lock, write_history
from fl_coordination._impl.fedpairs.common import json_write, save, load, moved, rng, set_rng
from .data import key_seed
from .engine import build_engine, LABELS


def one_round(engine, data, config, round_index):
    seed_all(key_seed(config['seed'],'training',round_index) % (2**32),config['deterministic'])
    record=engine.step(data.local_batches(round_index,engine.device))
    if record['client_ids']!=list(range(config['clients'])) or record['round']!=round_index:
        raise RuntimeError('Client/round mismatch')
    if [record['client_samples'][str(i)] for i in record['client_ids']]!=data.fit_sizes:
        raise RuntimeError('Round omitted or duplicated fitting samples')
    if [record['client_steps'][str(i)] for i in record['client_ids']]!=data.steps_per_client:
        raise RuntimeError('Local batch count mismatch')
    return record


def run_method(config,data,method,device,result_dir,weight_dir,manifest,
               *,engine_factory=build_engine,stop_after_round=None,stop_event=None,
               display_label=None,adaptation=None):
    label = display_label if display_label is not None else LABELS[method]
    rr,wr=Path(result_dir),Path(weight_dir)
    rr.mkdir(parents=True,exist_ok=True);wr.mkdir(parents=True,exist_ok=True)
    if rr.resolve().is_relative_to(wr.resolve()) or wr.resolve().is_relative_to(rr.resolve()):
        raise ValueError('Results and weights need separate roots')
    identity=fingerprint(dict(config=config,method=method,data=data.digest,sources=manifest['sha256']))
    budget=data.budget()
    with run_lock(wr/'.run.lock'):
        last_path,best_path=wr/'last.pt',wr/'best.pt'
        if (rr/'final.json').exists():
            final=json.loads((rr/'final.json').read_text())
            if final['identity_sha256']!=identity: raise RuntimeError('Completed identity changed')
            for p in [last_path,best_path,*[wr/f'round_{r:04d}.pt' for r in config['checkpoint_rounds']],
                      *[wr/f'best_until_{r:04d}.pt' for r in config['checkpoint_rounds']]]:
                if not p.exists(): raise RuntimeError('Missing required checkpoint: '+str(p))
            return final
        seed_all(config['seed'],config['deterministic']);engine=engine_factory(config,method,device)
        start,history,best,elapsed=0,[],None,0.
        if last_path.exists():
            state=load(last_path)
            if state['identity_sha256']!=identity: raise RuntimeError('Resume source/config/data identity changed')
            engine.load_state_dict(state['engine']);set_rng(state['rng'])
            start,history,best,elapsed=state['round'],state['history'],state['best'],state['elapsed_seconds']
            if len(history)!=start or engine.step_index!=start: raise RuntimeError('Resume round/history mismatch')
            stored=load(best_path) if best_path.exists() else None
            if stored is None or stored['identity_sha256']!=identity or stored['round']!=best['round']:
                if best['round']!=start: raise RuntimeError('Restore earlier selected checkpoint before resuming')
                save(best_path,dict(identity_sha256=identity,round=start,validation=best['validation'],model=moved(engine.model_state())))
            milestones(state,wr,config)
            for r in config['checkpoint_rounds']:
                if r<=start and (not (wr/f'round_{r:04d}.pt').exists() or not (wr/f'best_until_{r:04d}.pt').exists()):
                    raise RuntimeError('Missing earlier milestone')
            del state,stored
        elif best_path.exists() or (rr/'metrics.jsonl').exists():
            raise RuntimeError('Orphan outputs; refusing to overwrite')
        json_write(rr/'metadata.json',dict(identity_sha256=identity,method=method,label=label,
                   config=config,budget=budget,source_manifest=manifest,split_sha256=data.digest,
                   initial_model_sha256=engine.initial_hash,weights=str(wr),
                   selection='Highest mean client validation accuracy; earliest tie',
                   adaptation=adaptation or 'Author Retina ownership; patient holdouts and private heads; client-local BN buffers. '
                              'FedAvg: size-weighted SGD endpoints. Coordinated methods: A(mean trajectory gradients) + server Adam. '
                              'ASAM: direct A(mean p), no second h branch. FedAWA-Agg: released weight optimizer '
                              'with per-client eta*K_i endpoint context, applied to shared trajectory mean gradients.'))
        data.save_manifest(rr/'split_manifest.json');write_history(rr/'metrics.jsonl',history)
        print(f'{label} | {config["rounds"]} full-data Epochs | resume {start} | samples/round {sum(data.fit_sizes)}',flush=True)
        for r in range(start+1,config['rounds']+1):
            if stop_event is not None and stop_event.is_set(): return None
            begin=time.monotonic();record=one_round(engine,data,config,r)
            validation=engine.evaluate(data,'validation')
            elapsed+=time.monotonic()-begin
            improved=best is None or validation['mean']>best['validation']['mean']
            if improved: best=dict(round=r,validation=validation)
            record.update(validation=validation,elapsed_seconds=elapsed)
            history.append(record)
            state=dict(identity_sha256=identity,round=r,engine=moved(engine.state_dict()),
                       history=history,best=best,rng=rng(),elapsed_seconds=elapsed)
            save(last_path,state)
            if improved: save(best_path,dict(identity_sha256=identity,round=r,validation=validation,model=moved(engine.model_state())))
            milestones(state,wr,config);del state
            write_history(rr/'metrics.jsonl',history)
            print(f'{label} | round/data Epoch {r}/{config["rounds"]} | '
                  f'val mean {validation["mean"]:.2f}% | worst {validation["worst"]:.2f}% | selected {best["round"]}',flush=True)
            if stop_after_round is not None and r>=stop_after_round: return None
        if sum(x['samples'] for x in history)!=budget['training_sample_visits']:
            raise RuntimeError('Final data budget mismatch')
        final_test=engine.evaluate(data,'test')
        selected=load(best_path)
        if selected['identity_sha256']!=identity: raise RuntimeError('Selected identity mismatch')
        engine.load_model(selected['model'])
        selected_test=engine.evaluate(data,'test');public=engine.evaluate(data,'public_test')
        final=dict(identity_sha256=identity,method=method,display_method=label,
                   communication_rounds=engine.step_index,full_data_epochs=config['rounds'],seed=config['seed'],
                   selected_round=best['round'],selected_validation=best['validation'],
                   selected_test=selected_test,final_test=final_test,selected_public_test=public,
                   budget=budget,elapsed_seconds=elapsed,split_sha256=data.digest,
                   initial_model_sha256=engine.initial_hash,checkpoint_best=str(best_path),checkpoint_resume=str(last_path),
                   permanent_checkpoints={str(r):str(wr/f'round_{r:04d}.pt') for r in config['checkpoint_rounds']})
        json_write(rr/'final.json',final)
        print(f'DONE {label} | client test mean {selected_test["mean"]:.2f}% | worst {selected_test["worst"]:.2f}%',flush=True)
        return final
