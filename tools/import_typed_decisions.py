"""Convert the public typed-decisions dataset into decisionkit JSONL.

The dataset stores one row per case: ``state`` and ``questions`` are JSON
strings, and ``gold`` holds the teacher distribution per question. Its three
question types line up with the contract one to one:

* ``choice`` -> ``choice`` with ``options`` taken from the ordered criteria map;
* ``noul``   -> ``boolean`` (the true/false criteria);
* ``score``  -> ``score`` with ``levels`` taken from the ordered criteria list.

The official test split is written untouched. Because the dataset ships no
development split, every tenth training case is held out for calibration, and
the manifest records that decision.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

DATASET = 'LocalLLaMA/typed-decisions'
CONFIG = 'all'
ENDPOINT = 'https://datasets-server.huggingface.co/rows'
PAGE = 100
CALIBRATION_EVERY = 10
KIND_TO_PRIMITIVE = {'choice': 'choice', 'noul': 'boolean', 'score': 'score'}
RETRIES = 6
PAUSE_S = 0.6


def fetch_split(split: str, page: int = PAGE):
    """Yield every row of one split, following pagination and backing off on 429."""
    offset = 0
    while True:
        query = urllib.parse.urlencode({'dataset': DATASET, 'config': CONFIG,
                                        'split': split, 'offset': offset, 'length': page})
        request = urllib.request.Request(f'{ENDPOINT}?{query}', headers={'User-Agent': 'decisionkit'})
        payload = None
        for attempt in range(RETRIES):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.loads(response.read().decode('utf-8'))
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == RETRIES - 1:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == RETRIES - 1:
                    raise
            time.sleep(PAUSE_S * (2 ** attempt))
        rows = payload.get('rows', [])
        if not rows:
            return
        for entry in rows:
            yield entry['row']
        if payload.get('partial') is False and len(rows) < page:
            return
        offset += len(rows)
        time.sleep(PAUSE_S)


def convert_row(row: dict) -> tuple[dict | None, dict]:
    """Return (converted row or None, skip counters)."""
    questions = json.loads(row['questions'])
    gold = json.loads(row['gold'])
    prepared, labels = [], {}
    skipped = {}
    for key, question in questions.items():
        kind = question.get('type')
        if kind not in KIND_TO_PRIMITIVE:
            skipped[f'unknown_type:{kind}'] = skipped.get(f'unknown_type:{kind}', 0) + 1
            continue
        criteria = question.get('criteria')
        primitive = KIND_TO_PRIMITIVE[kind]
        body = {'id': key, 'type': primitive, 'question': question['instructions']}
        if primitive == 'choice':
            if not isinstance(criteria, dict) or len(criteria) < 2:
                skipped['choice_without_options'] = skipped.get('choice_without_options', 0) + 1
                continue
            body['options'] = dict(criteria)
            keys = list(criteria)
        elif primitive == 'boolean':
            # 600 of 1800 training prompts ship without true/false criteria; the
            # contract then labels the two candidates TRUE and FALSE.
            if criteria:
                body['criteria'] = {name: str(text) for name, text in criteria.items()}
            keys = ['true', 'false']
        else:
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                skipped['score_without_levels'] = skipped.get('score_without_levels', 0) + 1
                continue
            body['levels'] = list(criteria)
            keys = [str(index) for index in range(len(criteria))]
        distribution = gold.get(key, {}).get('probabilities')
        if not distribution or set(distribution) != set(keys):
            skipped['label_mismatch'] = skipped.get('label_mismatch', 0) + 1
            continue
        prepared.append(body)
        labels[key] = {name: float(value) for name, value in distribution.items()}
    if not prepared:
        skipped['empty_row'] = 1
        return None, skipped
    return ({'id': row['id'], 'group_id': row['id'], 'workflow': row.get('workflow'),
             'request': {'state': json.loads(row['state']), 'questions': prepared},
             'labels': labels, 'label_quality': 'public teacher distribution',
             'source': f'{DATASET} ({CONFIG})'}, skipped)


def convert_split(row_iter):
    rows, skipped = [], {}
    for row in row_iter:
        converted, row_skipped = convert_row(row)
        for key, count in row_skipped.items():
            skipped[key] = skipped.get(key, 0) + count
        if converted is not None:
            rows.append(converted)
    return rows, skipped


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                    encoding='utf-8')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=Path('data/typed-decisions'))
    parser.add_argument('--calibration-every', type=int, default=CALIBRATION_EVERY)
    args = parser.parse_args(argv)

    train, train_skipped = convert_split(fetch_split('train'))
    test, test_skipped = convert_split(fetch_split('test'))
    if not train or not test:
        print('the dataset server returned no rows', file=sys.stderr)
        return 1
    step = max(2, args.calibration_every)
    calibration = train[::step]
    keep = [row for index, row in enumerate(train) if index % step]
    counts = {'train': len(keep), 'calibration': len(calibration), 'test': len(test)}
    write_jsonl(args.out / 'train.jsonl', keep)
    write_jsonl(args.out / 'calibration.jsonl', calibration)
    write_jsonl(args.out / 'test.jsonl', test)
    questions = sum(len(row['request']['questions']) for row in keep + calibration + test)
    manifest = {'dataset': DATASET, 'config': CONFIG, 'endpoint': ENDPOINT,
                'counts': counts, 'questions': questions,
                'type_mapping': KIND_TO_PRIMITIVE,
                'calibration_split': f'every {step}th training case',
                'notes': ['Teacher distributions, not human ground truth.',
                          'The official test split is used as shipped; no case is shared with train.',
                          'Boolean prompts without true/false criteria keep the contract defaults.'],
                'skipped_questions': {'train': train_skipped, 'test': test_skipped}}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                            encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
