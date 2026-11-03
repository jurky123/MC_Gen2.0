"""Annotate the agreed existing-description subset with the same grounded protocol."""
import argparse
from pathlib import Path
import relabel_grounded_items as implementation
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=2);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--tag',required=True);a=p.parse_args();a.limit=0
 implementation.ROOT=Path('outputs/grounded_label_revision_10k_20261103');implementation.annotate(a)
