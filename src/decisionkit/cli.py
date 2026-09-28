"""Command line: synth, train, evaluate, serve, verify."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .contract import ContractError, prepare
from .features import question_features
from .metrics import latency, summarize
from .model import DecisionModel
from .synth import build, write_splits
from .train import load_model, load_rows, save_model, train

VERIFY_REQUEST = {'state': {'workflow': 'agent_completion',
                            'reviewer_notes': 'passes only the visible test',
                            'observed_public_tests': {'passed': True,
                                                      'notes': 'the last run reported FAILED'},
                            'next_command': 'pip install pandas'},
                  'questions': [
                      {'id': 'spec', 'type': 'boolean',
                       'question': 'Does the current implementation satisfy every stated requirement?',
                       'criteria': {'true': 'Every stated requirement, including edge cases, is met.',
                                    'false': 'At least one requirement is not met.'}},
                      {'id': 'next', 'type': 'choice',
                       'question': 'Which action should run next for this case?',
                       'options': {option: option for option in
                                   ['run the test suite', 'read the failing test',
                                    'patch the implementation', 'ask the user to clarify']}},
                      {'id': 'risk', 'type': 'score',
                       'question': 'How risky is the next command in a developer workspace?',
                       'levels': ['0 - read only', '1 - routine edit',
                                  '2 - package or configuration change',
                                  '3 - destructive or credential access']}]}


def load(path) -> DecisionModel:
    scorer, temperatures, meta = load_model(path)
    return DecisionModel(scorer, temperatures, meta)


def fit_temperatures(scorer, rows) -> dict:
    from .calibration import fit_temperature
    from .train import row_examples

    collected: dict[str, list] = {}
    for row in rows:
        request = prepare(row['request']).requests[0]
        for candidate_rows, target, question_id in row_examples(row, scorer.dim):
            kind = next(question.type for question in request.questions if question.id == question_id)
            collected.setdefault(kind, []).append((scorer.logits(candidate_rows), target))
    temperatures = {}
    for kind, pairs in collected.items():
        temperature, _ = fit_temperature(np.stack([pair[0] for pair in pairs]),
                                         np.stack([pair[1] for pair in pairs]))
        temperatures[kind] = temperature
    return temperatures


def _synth(args) -> int:
    rows = build(args.cases, args.seed)
    counts = write_splits(rows, args.out, args.validation, args.test)
    print(json.dumps({'cases': len(rows), 'splits': counts, 'out': str(args.out)}, indent=2))
    return 0


def _train(args) -> int:
    rows = load_rows(args.train)
    scorer, history = train(rows, epochs=args.epochs, learning_rate=args.learning_rate,
                            dim=args.dim, brier_weight=args.brier_weight, seed=args.seed,
                            progress=lambda entry: print(json.dumps(entry), flush=True))
    temperatures = fit_temperatures(scorer, load_rows(args.validation)) if args.validation else {}
    save_model(args.output, scorer, temperatures,
               meta={'name': args.name, 'train_rows': len(rows), 'epochs': args.epochs,
                     'dim': args.dim, 'brier_weight': args.brier_weight,
                     'final_train': history[-1] if history else None})
    print(json.dumps({'model': str(args.output), 'temperatures': temperatures}, indent=2))
    return 0


def _records(model: DecisionModel, rows, ignore_temperatures: bool):
    records, seconds = [], []
    for row in rows:
        saved = model.temperatures
        if ignore_temperatures:
            model.temperatures = {}
        response = model.evaluate(row['request'])
        model.temperatures = saved
        seconds.append(response['usage']['wall_ms'] / 1000)
        for answer in response['results'][0]['answers']:
            target = (row.get('labels') or {}).get(answer['id'])
            if not target:
                continue
            gold = max(target, key=target.get)
            if answer['type'] == 'boolean':
                prediction = 'true' if answer['value'] else 'false'
            elif answer['type'] == 'choice':
                prediction = answer['value']
            else:
                prediction = str(answer['level'])
            records.append({'id': answer['id'], 'type': answer['type'], 'gold': gold,
                            'prediction': prediction, 'probabilities': answer['distribution'],
                            'top_probability': answer['top_probability'],
                            'status': answer['status'], 'correct': gold == prediction})
    return records, seconds


def _evaluate(args) -> int:
    model = load(args.model)
    rows = load_rows(args.data)[:args.limit] if args.limit else load_rows(args.data)
    records, seconds = _records(model, rows, args.ignore_temperatures)
    summary = {'model': str(args.model), 'rows': len(rows), 'temperatures': model.temperatures,
               'metrics': summarize(records), 'latency': latency(seconds)}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return 0


def _serve(args) -> int:
    from .service import serve
    serve(load(args.model), args.host, args.port)
    return 0


def _verify(args) -> int:
    print(json.dumps(load(args.model).evaluate(VERIFY_REQUEST), indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='decisionkit', description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)

    synth = sub.add_parser('synth', help='write deterministic synthetic cases')
    synth.add_argument('--out', type=Path, default=Path('data/synthetic'))
    synth.add_argument('--cases', type=int, default=400)
    synth.add_argument('--validation', type=int, default=60)
    synth.add_argument('--test', type=int, default=60)
    synth.add_argument('--seed', type=int, default=7)
    synth.set_defaults(handler=_synth)

    train_cmd = sub.add_parser('train', help='fit the candidate scorer')
    train_cmd.add_argument('--train', type=Path, required=True)
    train_cmd.add_argument('--validation', type=Path)
    train_cmd.add_argument('--output', type=Path, required=True)
    train_cmd.add_argument('--epochs', type=int, default=16)
    train_cmd.add_argument('--learning-rate', type=float, default=0.02)
    train_cmd.add_argument('--dim', type=int, default=1 << 15)
    train_cmd.add_argument('--brier-weight', type=float, default=0.1)
    train_cmd.add_argument('--seed', type=int, default=0)
    train_cmd.add_argument('--name', default='decisionkit-linear')
    train_cmd.set_defaults(handler=_train)

    evaluate = sub.add_parser('evaluate', help='score a split and print metrics')
    evaluate.add_argument('--data', type=Path, required=True)
    evaluate.add_argument('--model', type=Path, required=True)
    evaluate.add_argument('--limit', type=int)
    evaluate.add_argument('--output', type=Path)
    evaluate.add_argument('--ignore-temperatures', action='store_true')
    evaluate.set_defaults(handler=_evaluate)

    serve_cmd = sub.add_parser('serve', help='run the loopback HTTP service')
    serve_cmd.add_argument('--model', type=Path, required=True)
    serve_cmd.add_argument('--host', default='127.0.0.1')
    serve_cmd.add_argument('--port', type=int, default=8765)
    serve_cmd.set_defaults(handler=_serve)

    verify = sub.add_parser('verify', help='print one full decision for a fixed example')
    verify.add_argument('--model', type=Path, required=True)
    verify.set_defaults(handler=_verify)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except ContractError as exc:
        print(f'contract error: {exc}', file=sys.stderr)
        return 2
