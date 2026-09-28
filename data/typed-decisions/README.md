# typed-decisions (vendored copy)

These JSONL files are a converted copy of a public dataset. They are here so the
repository can be evaluated without network access; nothing in them is original
work of this project.

| | |
| --- | --- |
| Dataset | [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) |
| License | Apache-2.0 (see [`LICENSE.txt`](LICENSE.txt)) |
| Revision at import | `f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8` |
| Config | `all` |
| Imported with | [`tools/import_typed_decisions.py`](../../tools/import_typed_decisions.py) |

## Files

| File | Cases | Questions | Bytes | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `train.jsonl` | 1080 | 5400 | 3188280 | `8b8ce77ed7eb5e9d88feed9bfcaef76dfd43945b35d0b642856a354b500499bd` |
| `calibration.jsonl` | 120 | 600 | 354424 | `8e9d390a58ef98b141e6e73e75c59c0077e18f9f9f8715aef3ea0d5d6f2b6dba` |
| `test.jsonl` | 400 | 2000 | 1179341 | `1e37028e6803108c89b147417039e81b6938c476e83c91ff9cdec7fe16e67ec8` |

`manifest.json` is written by the importer and carries the same hashes plus the
counts and the type mapping.

## How the copy was produced

1. The upstream `all` config is read through the Hugging Face datasets-server
   rows endpoint, 100 rows per page.
2. Its three question types map onto this project's contract one to one:
   `choice` → `choice` (`options` from the ordered criteria map),
   `noul` → `boolean` (`criteria` true/false),
   `score` → `score` (`levels` from the ordered criteria list).
3. Every question keeps the teacher's `probabilities` map as its label, so the
   labels are distributions, not one-hot classes.
4. The upstream test split is copied as shipped. The upstream data ships no
   development split, so every tenth training case is written to
   `calibration.jsonl` for temperature fitting and the rest to `train.jsonl`.
   No case appears in two files. `label_agreement`, `factors` and the raw
   parquet files are not carried over.
5. Nothing was relabelled, filtered or repaired: all 1600 cases and all 8000
   questions converted, none skipped.

Re-running the importer overwrites these files. If the upstream revision has
moved, the hashes above will change; the manifest records the revision that was
observed at import time.

## Limits carried over

The labels are teacher distributions, not human ground truth, and the upstream
card describes the content as synthetic. Agreement with that teacher is what the
numbers in the top-level README measure.
