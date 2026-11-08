"""Serve the native16 comparison gallery without exposing other project files."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


class GalleryHandler(SimpleHTTPRequestHandler):
    def allowed(self):
        name = unquote(urlsplit(self.path).path).lstrip('/')
        if name in ('', 'index.html', 'lossless_restore.png',
                    'compression_01.png', 'compression_02.png', 'compression_03.png',
                    'evaluation', 'evaluation/', 'evaluation/index.html',
                    'overfit', 'overfit/', 'overfit/index.html',
                    'recovery', 'recovery/', 'recovery/index.html',
                    'recovery-evaluation', 'recovery-evaluation/',
                    'recovery-evaluation/index.html',
                    'five-task', 'five-task/', 'five-task/index.html',
                    'five-task-evaluation', 'five-task-evaluation/',
                    'five-task-evaluation/index.html',
                    'five-task-diagnosis', 'five-task-diagnosis/',
                    'five-task-diagnosis/index.html'):
            permitted = True
        else:
            parts = PurePosixPath(name).parts
            permitted = (len(parts) == 3 and parts[0] == 'images'
                         and parts[1] in ('original', 'nearest', 'box', 'palette')
                         and parts[2].endswith('.png'))
            permitted |= (len(parts) == 3 and parts[0] == 'evaluation'
                          and parts[1] in ('inputs', 'native', 'panels')
                          and parts[2].endswith('.png'))
            permitted |= (len(parts) == 3 and parts[0] == 'overfit'
                          and parts[1] in ('inputs', 'generated', 'panels')
                          and parts[2].endswith('.png'))
            permitted |= (len(parts) == 3 and parts[0] == 'recovery'
                          and parts[1] == 'panels'
                          and parts[2].endswith('.png'))
            permitted |= (len(parts) == 3 and parts[0] == 'recovery-evaluation'
                          and parts[1] in ('inputs', 'native', 'panels')
                          and parts[2].endswith('.png'))
            permitted |= (len(parts) == 3 and parts[0] in ('five-task', 'five-task-evaluation')
                          and parts[1] in ('inputs', 'native', 'panels')
                          and parts[2].endswith('.png'))
            permitted |= (len(parts) == 3 and parts[0] == 'five-task-diagnosis'
                          and parts[1] == 'panels' and parts[2].endswith('.png'))
        root = Path(self.directory).resolve()
        return permitted and (root/name).resolve().is_relative_to(root)

    def do_GET(self):
        if not self.allowed():
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self):
        if not self.allowed():
            self.send_error(404)
            return
        super().do_HEAD()

    def list_directory(self, path):
        self.send_error(404)
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path,
                        default=Path('outputs/native16_data_review_20261107'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    root = args.directory.resolve()
    if not (root/'index.html').is_file() or not (root/'images').is_dir():
        parser.error('Prepare the native16 comparison gallery first')
    handler = partial(GalleryHandler, directory=str(root))
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f'Native16 comparison: http://{args.host}:{args.port}/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
