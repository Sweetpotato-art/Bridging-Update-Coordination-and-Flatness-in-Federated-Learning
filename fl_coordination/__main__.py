"""Run the experiment matrix or report paired statistics."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from .experiment import endpoint, location, read_json, train
from .recipes import DATASETS, MECHANISMS, NAMES, SEEDS, recipe, recipes


def parser():
    cli = argparse.ArgumentParser(prog='python -m fl_coordination')
    commands = cli.add_subparsers(dest='action', required=True)
    for action in ('list', 'train', 'summarize', 'prepare'):
        command = commands.add_parser(action)
        command.add_argument('dataset', choices=DATASETS)
        if action in ('prepare', 'train'):
            command.add_argument('--data-root', required=True, type=Path)
        if action in ('train','summarize'):
            command.add_argument('--results', type=Path, default=Path('runs/results'))
            command.add_argument('--coordinator', choices=NAMES, required=True)
            command.add_argument('--mechanism', choices=MECHANISMS, default='baseline')
        if action == 'train':
            command.add_argument('--seed', type=int, choices=SEEDS, required=True)
            command.add_argument('--weights', type=Path, default=Path('runs/checkpoints'))
            command.add_argument('--device', default='cuda')
    for action in ('sensitivity', 'branch-replay'):
        diagnostic = commands.add_parser(action)
        diagnostic.add_argument('--data-root', required=True, type=Path)
        diagnostic.add_argument('--run', required=True, type=Path)
        diagnostic.add_argument('--checkpoint', required=True, type=Path)
        diagnostic.add_argument('--device', default='cuda')
    return cli


def scores(args, mechanism):
    rows = []
    for seed in SEEDS:
        entry = recipe(args.dataset, args.coordinator, mechanism, seed)
        path = location(args.results,args.dataset,seed,args.coordinator,mechanism) / 'final.json'
        rows.append(endpoint(read_json(path),entry['endpoint']))
    return rows


def stats(values):
    return {'mean': mean(values),'sample_sd': stdev(values)}


def summarize(args):
    rows = scores(args, args.mechanism)
    answer = {'seeds': list(SEEDS),'mean_accuracy_pct':stats([r['mean'] for r in rows]),
              'worst_client_accuracy_pct':stats([r['worst'] for r in rows])}
    if args.dataset == 'retina':
        eligible = [r.get('mean_available_auroc') for r in rows]
        if all(value is not None for value in eligible):
            answer['macro_auroc_available_clients'] = stats(eligible)
            answer['auroc_eligible_clients_by_seed'] = {
                seed: row['auroc_defined_clients'] for seed, row in zip(SEEDS, rows)}
    if args.mechanism != 'baseline':
        baselines = scores(args,'baseline')
        answer['paired_gain_pp'] = stats([row['mean']-base['mean'] for row,base in zip(rows,baselines)])
    return answer


def prepare(args):
    from .experiment import module
    root = args.data_root.resolve()
    if args.dataset == 'fashion_mnist':
        return module('fedteam.data').ensure(root)
    if args.dataset == 'cifar100':
        return module('cifar_recipes.data').ensure(root)
    if args.dataset == 'office_home':
        module('fedoffice.prepare').prepare(root)
    else:
        prep = module('fedretina.prepare')
        prep.extract(prep.download(root), root/'unpacked')
        root = root/'unpacked'/'Retina'
    return {'data_root': str(root)}


def main():
    args = parser().parse_args()
    if args.action == 'list':
        result = {key:dict(endpoint=row['endpoint'], rho=row['config']['rho'])
                  for key,row in recipes(args.dataset).items()}
    elif args.action == 'prepare':
        result = prepare(args)
    elif args.action == 'summarize':
        result = summarize(args)
    elif args.action in ('sensitivity', 'branch-replay'):
        import torch
        torch.set_num_threads(4)
        device = torch.device(args.device)
        if args.action == 'sensitivity':
            from .sensitivity import probe
            result = probe('fashion', args.run, args.checkpoint, args.data_root, device)
        else:
            from .branch_replay import run
            result = run(args.run, args.checkpoint, args.data_root, device)
    else:
        import torch
        torch.set_num_threads(4)
        result = train(args.dataset, args.data_root, args.results, args.weights,
                       args.coordinator, args.mechanism, args.seed, torch.device(args.device))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
