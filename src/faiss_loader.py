from extract_embeddings import VGGishEmbeddingsModel
import tensorflow as tf
import numpy as np
import faiss
import lmdb
import json
import os

from tf_parser import parse_tfrecords_to_embeddings


AUDIOSET_INDEX_FILE = "audioset.index"
AUDIOSET_LMDB_FILE = "audioset.lmdb"


class FaissLoader():
    def __init__(self, dataset=None):
        # if the index file exists:
        if os.path.exists(AUDIOSET_INDEX_FILE) and os.path.exists(AUDIOSET_LMDB_FILE):
            print(f"Loading FAISS index from {AUDIOSET_INDEX_FILE}")
            self.index = faiss.read_index(AUDIOSET_INDEX_FILE)
        else:
            if dataset is not None:
                self.index = self._init_index(dataset)
            else:
                raise Exception("Dataset is None."
                                "You pass a valid dataset if the FAISS index is not stored in disk")

        print(f"Loading LMDB from {AUDIOSET_LMDB_FILE}")
        self.lmdb = lmdb.open(AUDIOSET_LMDB_FILE, readonly=True, lock=False)

    def _init_index(self, dataset):
        embedding_length = 128
        index = faiss.IndexFlatL2(embedding_length)

        emb_buffer = []
        next_id = 0
        buffer_size = 2000  # adjust depending on RAM

        env = lmdb.open(AUDIOSET_LMDB_FILE, map_size=3 *
                        (1024**3))  # 3GB map size

        with env.begin(write=True) as txn:
            for embeddings, context in dataset:
                arr = embeddings.numpy().astype("float32")
                num_steps = arr.shape[0]

                video_id = context["video_id"].numpy().decode("utf-8")
                start_sec = context["start_time_seconds"].numpy()
                end_sec = context["end_time_seconds"].numpy()
                labels = tf.sparse.to_dense(context["labels"]).numpy().tolist()

                times = np.linspace(start_sec, end_sec,
                                    num=num_steps, endpoint=False)

                emb_buffer.append(arr)

                for t in times:
                    meta = {
                        "video_id": video_id,
                        "timestamp": float(t),
                        "labels": labels,
                    }
                    txn.put(str(next_id).encode(), json.dumps(meta).encode())
                    next_id += 1

                if sum(e.shape[0] for e in emb_buffer) >= buffer_size:
                    chunk = np.vstack(emb_buffer)
                    index.add(chunk)
                    emb_buffer.clear()

            if emb_buffer:
                chunk = np.vstack(emb_buffer)
                index.add(chunk)
                emb_buffer.clear()

        print("Final index size:", index.ntotal)

        # Save FAISS index to disk
        faiss.write_index(index, AUDIOSET_INDEX_FILE)
        print(f"FAISS index saved as {AUDIOSET_INDEX_FILE}")
        return index

    def get_k_nearest_neighbors(self, embeddings, k=1):
        distances, indices = self.index.search(embeddings, k)

        with self.lmdb.begin() as txn:
            for rank, (index, distance) in enumerate(zip(indices[0], distances[0])):
                raw = txn.get(str(int(index)).encode())
                if raw:
                    meta = json.loads(raw.decode())
                    print(f"Rank {rank}: video={meta['video_id']}, "
                          f"time={meta['timestamp']:.1f}s, "
                          f"dist={distance:.4f}, labels={meta['labels']}")


def main():
    print("Parsing TFRecords to embeddings")
    embedding_dataset = parse_tfrecords_to_embeddings(
        'audioset_v1_embeddings/bal_train/__.tfrecord', True)

    model = VGGishEmbeddingsModel()
    embeddings = model.get_embeddings('assets/YouDoSomethingToMe2.mp3', True)
    tf.keras.backend.clear_session()

    faiss_loader = FaissLoader(embedding_dataset)
    faiss_loader.get_k_nearest_neighbors(embeddings, k=3)


if __name__ == "__main__":
    main()
