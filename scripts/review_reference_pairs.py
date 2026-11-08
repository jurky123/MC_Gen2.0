"""Run a local, durable reference-pair selection website without loading a model."""
import argparse
from pathlib import Path

from mcgen.pair_review import create_review_app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, default=Path('outputs/spatial_reference_20261104/pairs.json'))
    parser.add_argument('--state-dir', type=Path, default=Path('outputs/pair_review'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    import uvicorn
    app = create_review_app(args.pairs, args.state_dir)
    summary = app.state.review.summary()
    print(f'MC Pair Studio: http://{args.host}:{args.port}', flush=True)
    print(f'{summary["targets"]} targets / {summary["pairs"]} pairs; decisions: {args.state_dir.resolve()}', flush=True)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
