"""Chunked, pipelined text encoding for the training loop.

The frozen text tower (GPU1) is launch-bound: encoding 256 short prompts takes
~0.6 s regardless of length. Encoding the whole accumulation window's prompts
in ONE forward and running it on a background thread (while the DiT trains on
GPU0) cuts step time roughly in half without changing any training semantics:
same batches, same order, same loss composition. Only the *text encode* is
batched/pipelined; the DiT still sees one micro-batch at a time.
"""
import queue
import threading
class EncodeWorkerError(RuntimeError):
    pass


class ChunkedEncodedLoader:
    """Yield chunks of up to ``accum`` micro-batches with text pre-encoded.

    Text encoding happens on a background thread: chunk k+1's prompts are
    encoded on the tower device while the main thread runs DiT fwd/bwd for
    chunk k. Each yielded item is (x, text_pair_or_text, aux) where
    ``text_pair`` is a (hidden, mask) tensor pair already moved to the main
    device.
    """

    def __init__(self, loader, text_encoder, accum, tower_device, main_device,
                 queue_size=2):
        self.loader = loader
        self.encoder = text_encoder
        self.accum = max(1, accum)
        self.tower_device = tower_device
        self.main_device = main_device
        self._q = queue.Queue(maxsize=queue_size)
        self._err = None
        self._stop = threading.Event()
        self._thread = None

    def _worker(self):
        try:
            batch_iter = iter(self.loader)
            done = False
            while not done:
                chunk = []
                for _ in range(self.accum):
                    try:
                        chunk.append(next(batch_iter))
                    except StopIteration:
                        done = True
                        break
                if not chunk:
                    break
                prompts = []
                for b in chunk:
                    prompts.extend(b[1])
                hidden, mask = self.encoder.encode(prompts)
                pairs, offset = [], 0
                for b in chunk:
                    n = len(b[1])
                    pairs.append((hidden[offset:offset + n], mask[offset:offset + n]))
                    offset += n
                self._put((chunk, pairs))
        except BaseException as exc:  # propagate to the consumer
            self._err = exc
        finally:
            try:
                self._q.put(None, timeout=5)
            except Exception:
                pass

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=1.0)
                return
            except queue.Full:
                continue
        self._q = None

    def __iter__(self):
        # Fresh event per epoch: the previous iteration's finally-clause set it.
        self._stop = threading.Event()
        self._err = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        try:
            while not self._stop.is_set():
                item = self._q.get()
                if item is None:
                    if self._err is not None:
                        raise self._err
                    break
                chunk, pairs = item
                out = []
                for b, (hidden, mask) in zip(chunk, pairs):
                    aux = b[2] if len(b) > 2 else None
                    out.append((b[0], (hidden.to(self.main_device, non_blocking=True),
                                       mask.to(self.main_device, non_blocking=True)), aux))
                yield out
        finally:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=60)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=60)
