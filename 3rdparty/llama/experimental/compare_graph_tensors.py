"""Compare offline ggml graph captures without changing acceptance criteria.

The interposer changes allocation lifetimes. Public final-result invariance
must be checked separately before attributing any intermediate difference.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_graph(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    result = {}
    for row in rows:
        key = int(row["index"])
        if key in result:
            raise ValueError(f"duplicate captured node: {path}:{key}")
        filename = Path(row["file"])
        if filename.name != str(filename):
            raise ValueError("capture files must be local basenames")
        shape = tuple(int(row[f"ne{i}"]) for i in range(4))
        dtype = {"f32": "<f4", "f16": "<f2"}[row["type"]]
        value = np.fromfile(path.parent / filename, dtype=dtype)
        if value.size != int(np.prod(shape)) or not np.isfinite(value).all():
            raise ValueError(f"invalid captured tensor: {path}:{key}")
        result[key] = (row, value)
    return result


def compare(baseline, candidate):
    before = {path.name: path for path in baseline.glob("graph-*.tsv")
              if path.stem.removeprefix("graph-").isdigit()}
    after = {path.name: path for path in candidate.glob("graph-*.tsv")
             if path.stem.removeprefix("graph-").isdigit()}
    if not before or before.keys() != after.keys():
        raise ValueError("captured graph inventory differs or is empty")
    report = {"scope": "offline intermediate diagnosis only; not release acceptance",
              "final_result_invariance_checked": False, "graphs": []}
    fields = ("name", "op", "type", "ne0", "ne1", "ne2", "ne3")
    for name in sorted(before, key=lambda value: int(value.split("-")[1].split(".")[0])):
        left, right = read_graph(before[name]), read_graph(after[name])
        if left.keys() != right.keys():
            raise ValueError(f"captured node inventory differs: {name}")
        differences = []
        for index in sorted(left):
            metadata, x = left[index]
            other, y = right[index]
            if any(metadata[field] != other[field] for field in fields):
                raise ValueError(f"graph node identity differs: {name}:{index}")
            error = np.abs(x.astype(np.float64) - y.astype(np.float64))
            changed = int(np.count_nonzero(x != y))
            if changed:
                differences.append(dict(index=index, name=metadata["name"], op=metadata["op"],
                                        shape=[int(metadata[f"ne{i}"]) for i in range(4)],
                                        changed=changed, elements=int(x.size),
                                        max_abs=float(error.max()), mae=float(error.mean())))
        report["graphs"].append(dict(graph=name, captured_nodes=len(left),
                                      changed_nodes=len(differences),
                                      first_difference=differences[0] if differences else None,
                                      differences=differences))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"graphs": [{key: value for key, value in row.items() if key != "differences"}
                                  for row in report["graphs"]]}, indent=2))


if __name__ == "__main__":
    main()
