"""Stable pair identities and reviewed selections without reindexing cached features."""
import hashlib
import json
from pathlib import Path

from .catalog import file_hash

FORMAT = 'mcgen.reviewed-pairs.v1'


def pair_id(row):
    values = [row['target_hash'], row['reference_sha256'], row['reference_generator']]
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def selected_indices(selection_file, pairs_file, rows):
    selection = json.loads(Path(selection_file).read_text())
    if selection.get('format') != FORMAT:
        raise ValueError('Unsupported reviewed selection format')
    if selection.get('source_pairs_sha256') != file_hash(pairs_file):
        raise ValueError('Selection belongs to a different source pair manifest')
    indices = selection.get('selected_indices')
    if (not isinstance(indices, list) or not indices
            or any(type(i) is not int or not 0 <= i < len(rows) for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('Selection indices must be distinct valid source rows')
    chosen = set(indices)
    if not any(rows[i]['split'] == 'train' for i in chosen):
        raise ValueError('No reviewed training pairs selected')
    if not any(rows[i]['split'] == 'val' for i in chosen):
        raise ValueError('No validation pairs selected')
    if any(rows[i]['split'] not in ('train', 'val') for i in chosen):
        raise ValueError('Reviewed training selection cannot include test samples')
    expected_ids = selection.get('selected_pair_ids')
    if expected_ids is not None and expected_ids != [pair_id(rows[i]) for i in indices]:
        raise ValueError('Selection pair identities differ from original cache rows')
    return chosen
