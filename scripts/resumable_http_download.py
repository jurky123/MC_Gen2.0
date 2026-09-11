"""Strict HTTP Range downloader that never truncates a valid partial file."""
import argparse
import socket
import time
import re
import urllib.error
import urllib.request
from pathlib import Path


def download(url: str, output: Path, proxy: str, retries: int = 100) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    proxy_handler = urllib.request.ProxyHandler(
        {"http": proxy, "https": proxy} if proxy else {}
    )
    opener = urllib.request.build_opener(proxy_handler)
    for attempt in range(1, retries + 1):
        offset = output.stat().st_size if output.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            request = urllib.request.Request(url, headers=headers)
            with opener.open(request, timeout=60) as response:
                status = response.status
                content_range = response.headers.get("Content-Range", "")
                if offset:
                    expected = f"bytes {offset}-"
                    if status != 206 or not content_range.startswith(expected):
                        raise RuntimeError(
                            f"server rejected resume: status={status} "
                            f"Content-Range={content_range!r} offset={offset}"
                        )
                match = re.search(r"/(\d+)$", content_range)
                if match:
                    expected_total = int(match.group(1))
                else:
                    content_length = int(response.headers.get("Content-Length", "0"))
                    expected_total = offset + content_length if content_length else 0
                mode = "ab" if offset else "wb"
                with output.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                actual = output.stat().st_size
                if expected_total and actual != expected_total:
                    raise OSError(f"incomplete response: got={actual} expected={expected_total}")
                print(f"complete path={output} bytes={output.stat().st_size}", flush=True)
                return
        except urllib.error.HTTPError as exc:
            content_range = exc.headers.get("Content-Range", "")
            match = re.search(r"\*/(\d+)$", content_range)
            if exc.code == 416 and match and int(match.group(1)) == offset:
                print(f"already complete path={output} bytes={offset}", flush=True)
                return
            print(f"retry={attempt}/{retries} offset={offset} error={exc}", flush=True)
            if attempt == retries:
                raise
            time.sleep(5)
        except (urllib.error.URLError, socket.timeout, OSError, RuntimeError) as exc:
            print(f"retry={attempt}/{retries} offset={offset} error={exc}", flush=True)
            if attempt == retries:
                raise
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--proxy", default="http://127.0.0.1:10808")
    parser.add_argument("--retries", type=int, default=100)
    args = parser.parse_args()
    download(args.url, args.output, args.proxy, args.retries)
