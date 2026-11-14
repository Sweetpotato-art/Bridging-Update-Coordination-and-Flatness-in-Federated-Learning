"""Full data epochs; validation-only selection; permanent 100/200/300 checkpoints."""
import csv
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch

from fl_coordination._impl.fedmini.data import fingerprint
from fl_coordination._impl.fedmini.runtime import seed_all, run_lock, write_history
from fl_coordination._impl.fedpairs.common import json_write, save, load, moved, rng, set_rng
from fl_coordination._impl.fedseries.runtime import bind_experiment, epoch_summary
from .data import epoch_plan, plan_summary, DOMAINS
from .engine import build_engine, LABELS


def permanent_checkpoints(state, weight_dir, milestones):
    """Fixed-epoch files contain optimizer/coordinator/head/BN/RNG state, not just weights."""
    epoch=state['epoch']
    if epoch not in milestones:
        return
    destination=weight_dir/f'epoch_{epoch:03d}.pt'
    if not destination.exists():
        save(destination,state)
    else:
        old=load(destination)
        if old['identity_sha256']!=state['identity_sha256'] or old['epoch']!=epoch:
            raise RuntimeError('Conflicting permanent checkpoint')
        del old
    selected=weight_dir/f'best_until_{epoch:03d}.pt'
    if not selected.exists():
        temp=selected.with_suffix('.pt.tmp')
        shutil.copyfile(weight_dir/'best.pt',temp)
        temp.replace(selected)


def write_curves(root, history):
    write_history(root/'metrics.jsonl',history)
    temporary=root/'convergence.csv.tmp'
    with temporary.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream)
        writer.writerow(['epoch','round','train_loss','validation_mean','validation_worst',
                         *['validation_'+d for d in DOMAINS],'seconds_total'])
        for row in history:
            val=row['validation']
            writer.writerow([row['epoch'],row['round'],np.mean(row['train_loss']),val['mean'],val['worst'],
                             *[v['accuracy'] for v in val['clients']],row['elapsed_seconds']])
    temporary.replace(root/'convergence.csv')


def one_epoch(engine, data, cfg, epoch, *, verbose=True):
    records=[]
    planned=plan_summary(data,cfg,epoch)
    for index,selected in enumerate(epoch_plan(data,cfg,epoch),1):
        batches={cid:[data.batch('fit',ids,engine.device,epoch=epoch,cid=cid) for ids in chunks]
                 for cid,chunks in selected.items()}
        record=engine.step(batches)
        records.append(record)
        if verbose and (index%cfg['log_every_rounds']==0 or index==planned['rounds']):
            print(f'Epoch {epoch+1}/{cfg["epochs"]} | round {index}/{planned["rounds"]} | '
                  f'CE {np.mean(record["losses"]):.4f}',flush=True)
    summary=epoch_summary(records,cfg['clients'])
    if summary['client_samples']!=data.fit_sizes or summary['client_steps']!=data.steps_per_client:
        raise RuntimeError('Epoch did not exactly cover fitting data')
    return summary,planned


def run_method(cfg,data,method,device,result_dir,weight_dir,manifest,
               *,engine_factory=build_engine,stop_after_epoch=None):
    rr,wr=Path(result_dir),Path(weight_dir)
    rr.mkdir(parents=True,exist_ok=True)
    wr.mkdir(parents=True,exist_ok=True)
    if wr.resolve().is_relative_to(rr.resolve()) or rr.resolve().is_relative_to(wr.resolve()):
        raise ValueError('Results and weights must have separate roots')
    experiment=bind_experiment(cfg,data,manifest,rr)
    identity=fingerprint(dict(experiment=experiment['sha256'],method=method))
    budget=plan_summary(data,cfg)
    last_path,best_path=wr/'last.pt',wr/'best.pt'
    with run_lock(wr/'.run.lock'):
        if (rr/'final.json').exists():
            final=json.loads((rr/'final.json').read_text(encoding='utf-8'))
            if final['identity_sha256']!=identity:
                raise RuntimeError('Completed identity changed')
            for path in [last_path,best_path,
                         *[wr/f'epoch_{e:03d}.pt' for e in cfg['checkpoint_epochs']],
                         *[wr/f'best_until_{e:03d}.pt' for e in cfg['checkpoint_epochs']]]:
                if not path.exists():
                    raise RuntimeError('Missing required checkpoint: '+str(path))
            print('Already complete; preserving results.',flush=True)
            return final
        seed_all(cfg['seed'],cfg['deterministic'])
        engine=engine_factory(cfg,method,device)
        start,history,best,elapsed=0,[],None,0.
        if last_path.exists():
            state=load(last_path)
            if state['identity_sha256']!=identity:
                raise RuntimeError('Resume source/config/data identity changed')
            engine.load_state_dict(state['engine'])
            set_rng(state['rng'])
            start,history,best,elapsed=state['epoch'],state['history'],state['best'],state['elapsed_seconds']
            if len(history)!=start or engine.step_index!=start*budget['rounds']:
                raise RuntimeError('Resume history/round mismatch')
            stored=load(best_path) if best_path.exists() else None
            if stored is None or stored['identity_sha256']!=identity or stored['epoch']!=best['epoch']:
                if best['epoch']!=start:
                    raise RuntimeError('Earlier selected checkpoint must be restored')
                save(best_path,dict(identity_sha256=identity,epoch=start,validation=best['validation'],model=moved(engine.model_state())))
            del stored
            permanent_checkpoints(state,wr,cfg['checkpoint_epochs'])
            for e in cfg['checkpoint_epochs']:
                if e<=start and (not (wr/f'epoch_{e:03d}.pt').exists() or
                                 not (wr/f'best_until_{e:03d}.pt').exists()):
                    raise RuntimeError('An earlier permanent checkpoint is missing')
            del state
        elif (rr/'metrics.jsonl').exists() or best_path.exists():
            raise RuntimeError('Orphan logs/weights; refusing to overwrite')
        metadata=dict(method=method,display_method=LABELS[method],identity_sha256=identity,config=cfg,
                      sources=manifest,split_sha256=data.digest,initial_model_sha256=engine.initial_hash,
                      epoch_budget=budget,total_rounds=budget['rounds']*cfg['epochs'],
                      total_samples=data.samples_per_epoch*cfg['epochs'],weights=str(wr.resolve()),
                      selection='Highest arithmetic mean validation accuracy; earliest tie',
                      perturbation='ASAM, joint shared+private norm' if method=='fedfv_asam' else 'none',
                      coordination='A(mean_k p_ik), direct' if method=='fedfv_asam' else method,
                      domains=list(DOMAINS),evaluation='Held-out samples from participating domains')
        json_write(rr/'metadata.json',metadata)
        data.save_manifest(rr/'split_manifest.json')
        write_curves(rr,history)
        print(f'{LABELS[method]} | seed {cfg["seed"]} | {cfg["epochs"]} data Epochs | '
              f'{budget["rounds"]} rounds/Epoch | resume {start}',flush=True)
        for epoch in range(start,cfg['epochs']):
            begin=time.monotonic()
            summary,planned=one_epoch(engine,data,cfg,epoch)
            validation=engine.evaluate(data,'validation')
            elapsed+=time.monotonic()-begin
            history.append(dict(epoch=epoch+1,round=engine.step_index,validation=validation,
                                elapsed_seconds=elapsed,schedule_sha256=planned['sha256'],**summary))
            improved=best is None or validation['mean']>best['validation']['mean']
            if improved:
                best=dict(epoch=epoch+1,validation=validation)
            state=dict(identity_sha256=identity,epoch=epoch+1,engine=moved(engine.state_dict()),
                       history=history,best=best,rng=rng(),elapsed_seconds=elapsed)
            save(last_path,state)
            if improved:
                save(best_path,dict(identity_sha256=identity,epoch=epoch+1,
                                    validation=validation,model=moved(engine.model_state())))
            permanent_checkpoints(state,wr,cfg['checkpoint_epochs'])
            del state
            write_curves(rr,history)
            val_text=' | '.join(f'{d} {v["accuracy"]:.2f}%' for d,v in zip(DOMAINS,validation['clients']))
            print(f'{LABELS[method]} | Epoch {epoch+1} | {val_text} | mean {validation["mean"]:.2f}% '
                  f'| worst {validation["worst"]:.2f}% | selected {best["epoch"]}',flush=True)
            if epoch+1 in cfg['checkpoint_epochs']:
                json_write(rr/f'epoch_{epoch+1:03d}.json',dict(epoch=epoch+1,validation=validation,
                           best_so_far=best,checkpoint=str(wr/f'epoch_{epoch+1:03d}.pt')))
            if stop_after_epoch is not None and epoch+1>=stop_after_epoch:
                return None
        final_test=engine.evaluate(data,'test')
        selected=load(best_path)
        if selected['identity_sha256']!=identity:
            raise RuntimeError('Selected checkpoint identity differs')
        engine.load_model(selected['model'])
        selected_test=engine.evaluate(data,'test')
        final=dict(identity_sha256=identity,method=method,display_method=LABELS[method],seed=cfg['seed'],
                   epochs=cfg['epochs'],selected_epoch=best['epoch'],selected_validation=best['validation'],
                   selected_test=selected_test,final_test=final_test,initial_model_sha256=engine.initial_hash,
                   elapsed_seconds=elapsed,communication_rounds=engine.step_index,
                   training_samples=sum(r['samples'] for r in history),
                   checkpoint_best=str(best_path),checkpoint_resume=str(last_path),
                   permanent_checkpoints={str(e):str(wr/f'epoch_{e:03d}.pt') for e in cfg['checkpoint_epochs']})
        if (final['training_samples']!=metadata['total_samples'] or
                final['communication_rounds']!=metadata['total_rounds']):
            raise RuntimeError('Incorrect final sample budget')
        json_write(rr/'final.json',final)
        print(f'DONE {LABELS[method]} | selected test {selected_test["mean"]:.2f}% | worst {selected_test["worst"]:.2f}%',flush=True)
        return final
