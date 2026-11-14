"""Three endpoints from one unchanged 100-Epoch Retina trajectory."""
import json,signal,time
from pathlib import Path
from fl_coordination._impl.fedmini.data import fingerprint
from fl_coordination._impl.fedmini.runtime import seed_all,run_lock,write_history
from fl_coordination._impl.fedpairs.common import json_write,save,load,moved,rng,set_rng,preserve_rng
from fl_coordination._impl.fedretina.runtime import one_round
from .config import JOBS,settings
from .engine import build_engine
from .observer import Observer


def run_method(cfg,data,job,device,result_dir,weight_dir,manifest,reference,*,engine_factory=build_engine,stop_after_epoch=None,detector_settings=None,stop_event=None):
    rr,wr=Path(result_dir),Path(weight_dir)
    if wr.resolve().is_relative_to(rr.resolve()) or rr.resolve().is_relative_to(wr.resolve()):raise ValueError('Results and weights must be separate')
    rr.mkdir(parents=True,exist_ok=True);wr.mkdir(parents=True,exist_ok=True)
    grid=detector_settings if detector_settings is not None else settings()
    identity=fingerprint(dict(config=cfg,method=job,data=data.digest,source=manifest['sha256'],reference=reference,detectors=grid))
    with run_lock(wr/'.run.lock'):
        if (rr/'final.json').exists():
            f=json.loads((rr/'final.json').read_text())
            if f['identity_sha256']!=identity or not all(Path(p).is_file() for p in f['checkpoints'].values()):raise RuntimeError('Completed identity/weights changed')
            return f
        seed_all(cfg['seed'],cfg['deterministic']);engine=engine_factory(cfg,job,device)
        saved=load(wr/'last.pt') if (wr/'last.pt').exists() else None
        if saved is None and (rr/'metrics.jsonl').exists():raise RuntimeError('Orphan metrics; no overwrite')
        if saved and saved['identity_sha256']!=identity:raise RuntimeError('Resume identity changed')
        observer=Observer(engine,data,reference['validation'],grid,initial_losses=saved['observer']['initial_losses'] if saved else None)
        start,history,best,elapsed=0,[],None,0.
        if saved:
            engine.load_state_dict(saved['engine']);observer.load_state_dict(saved['observer'])
            start,history,best,elapsed=(saved[k] for k in ('epoch','history','best','elapsed_seconds'))
            set_rng(saved['rng'])
            if len(history)!=start or engine.step_index!=start or observer.grid.last_step!=start or start>cfg['rounds']:raise RuntimeError('Invalid resume boundary')
            del saved
        json_write(rr/'metadata.json',dict(identity_sha256=identity,method=job,combination=JOBS[job][0],rho=cfg['rho'],config=cfg,sources=manifest,split_sha256=data.digest,initial_model_sha256=engine.initial_hash,weights=str(wr),reference=reference,budget=data.budget(),detector_settings=grid,
            sampling='one coherent shared/private/BN state per completed server round; one round is one full-data Epoch',
            endpoints=['raw_fixed_epoch100','raw_highest_mean_validation_earliest_tie','weight_refinement_baseline_retention'],
            adaptation='Five-client weight refinement applied to global FL iterates. Private heads remain paired with their own BN; local temporary perturbations never sampled.'))
        data.save_manifest(rr/'split_manifest.json');write_history(rr/'metrics.jsonl',history)
        requested=[]
        def stop(sig,frame):requested.append(sig);print('Saving at full-Epoch boundary before stopping.',flush=True)
        handlers={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            for epoch in range(start+1,cfg['rounds']+1):
                if stop_event is not None and stop_event.is_set():return None
                begin=time.monotonic();observer.begin_epoch()
                rec=one_round(engine,data,cfg,epoch);observer.on_round(engine)
                val=engine.evaluate(data,'validation')
                if best is None or val['mean']>best['validation']['mean']:
                    best=dict(epoch=epoch,validation=val,model=moved(engine.model_state()))
                dense=observer.end_epoch(epoch);elapsed+=time.monotonic()-begin
                rec.update(epoch=epoch,validation=val,dense_validation=dense,elapsed_seconds=elapsed)
                history.append(rec)
                sb=time.monotonic()
                save(wr/'last.pt',dict(identity_sha256=identity,epoch=epoch,engine=moved(engine.state_dict()),observer=observer.state_dict(),history=history,best=best,elapsed_seconds=elapsed,rng=rng()))
                save_seconds=time.monotonic()-sb
                write_history(rr/'metrics.jsonl',history);json_write(rr/'observer_audit.json',observer.audit())
                json_write(rr/'progress.json',dict(epoch=epoch,epochs=cfg['rounds'],round=engine.step_index,elapsed_seconds=elapsed,last_epoch_wall_seconds=time.monotonic()-begin,checkpoint_seconds=save_seconds,candidates=observer.timing['candidates']))
                print(f'{job} Epoch {epoch}/{cfg["rounds"]} | raw val {val["mean"]:.2f} worst {val["worst"]:.2f} | valleys {observer.timing["candidates"]}',flush=True)
                if requested or (stop_event is not None and stop_event.is_set()):return None
                if stop_after_epoch is not None and epoch>=stop_after_epoch:return None
        finally:
            for s,h in handlers.items():signal.signal(s,h)
        if engine.step_index!=cfg['rounds'] or len(history)!=cfg['rounds'] or sum(r['samples'] for r in history)!=cfg['rounds']*sum(data.fit_sizes):raise RuntimeError('Incomplete full-data budget')
        chosen=observer.finalize()
        paths=dict(resume=str(wr/'last.pt'),raw=str(wr/f'raw_epoch_{cfg["rounds"]:03d}.pt'),raw_validation_selected=str(wr/'raw_validation_selected.pt'))
        save(paths['raw'],dict(identity_sha256=identity,epoch=cfg['rounds'],model=moved(engine.model_state()),validation=history[-1]['validation']))
        save(paths['raw_validation_selected'],dict(identity_sha256=identity,**best))
        selection=dict(identity_sha256=identity,budget_epoch=cfg['rounds'],reference=reference,raw_validation_rule='highest_mean_client_validation_accuracy_earliest_tie',raw_validation_selected={k:v for k,v in best.items() if k!='model'},
            weight_refinement_rule='min(worst_positive_relative_drop,mean_positive_relative_drop,mean_signed_relative_drop,candidate_id)',weight_refinement={k:v for k,v in chosen.items() if k!='state'} if chosen else None,weight_refinement_status='selected' if chosen else 'no_eligible_valley')
        if chosen:
            paths['weight_refinement']=str(wr/'weight_refinement_retention.pt');save(paths['weight_refinement'],dict(identity_sha256=identity,model=chosen['state'],**selection['weight_refinement']))
        json_write(rr/'selection.json',selection);json_write(rr/'observer_audit.json',observer.audit())
        def evaluate(model,val,baseline_endpoint):
            observer.scratch.load_model(model)
            t=observer.scratch.evaluate(data,'test');p=observer.scratch.evaluate(data,'public_test')
            base=reference['fixed_epoch100']['test'] if baseline_endpoint=='fixed_epoch100' else reference['test']
            return dict(test=t,public_test=p,validation=val,baseline_endpoint=baseline_endpoint,improvement_pp=t['mean']-base['mean'],improvement_vs_selected_baseline_pp=t['mean']-reference['test']['mean'],improvement_vs_fixed_baseline_pp=t['mean']-reference['fixed_epoch100']['test']['mean'])
        with preserve_rng():
            raw=evaluate(engine.model_state(),history[-1]['validation'],'fixed_epoch100')
            selected=evaluate(best['model'],best['validation'],'validation_selected_within_100');selected['epoch']=best['epoch']
            refinement=evaluate(chosen['state'],chosen['validation'],'validation_selected_within_100') if chosen else None
        final=dict(identity_sha256=identity,method=job,combination=JOBS[job][0],rho=cfg['rho'],seed=cfg['seed'],epochs=cfg['rounds'],full_data_epochs=cfg['rounds'],communication_rounds=engine.step_index,training_samples=sum(r['samples'] for r in history),elapsed_seconds=elapsed,raw=raw,raw_validation_selected=selected,weight_refinement=refinement,weight_refinement_status=selection['weight_refinement_status'],checkpoints=paths,split_sha256=data.digest,initial_model_sha256=engine.initial_hash)
        json_write(rr/'final.json',final)
        save(wr/'last.pt',dict(identity_sha256=identity,epoch=cfg['rounds'],engine=moved(engine.state_dict()),history=history,rng=rng(),elapsed_seconds=elapsed,completed=True,selection=selection,observer_summary=observer.audit()))
        print(f'DONE {job} raw100 {raw["test"]["mean"]:.2f} raw-selected {selected["test"]["mean"]:.2f}; weight refinement '+(f'{refinement["test"]["mean"]:.2f}' if refinement else 'no eligible valley'),flush=True)
        return final
