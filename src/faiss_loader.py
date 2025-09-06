import pickle
import random
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
                self._init_index(dataset)
                self.index = faiss.read_index(AUDIOSET_INDEX_FILE)
            else:
                raise Exception("Dataset is None."
                                "You must pass a valid dataset if the FAISS index is not stored in disk")

        print(f"Loading LMDB from {AUDIOSET_LMDB_FILE}")
        self.lmdb = lmdb.open(AUDIOSET_LMDB_FILE, readonly=True, lock=False)

    def _init_index(self, dataset):
        embedding_length = 128
        nlist, m, nbits = 4096, 16, 8

        print("Getting training sample")
        train_sample = self.sample_embeddings_for_training(
            dataset, num_samples=159745)

        print("Training index")
        quantizer = faiss.IndexFlatL2(embedding_length)
        index = faiss.IndexIVFPQ(quantizer, embedding_length, nlist, m, nbits)
        index.train(train_sample)

        print("Adding all embeddings to index")
        batch_size = 10000
        buffer_embeddings = []
        buffer_metadata = []
        row_id = 0

        lmdb_index = lmdb.open(AUDIOSET_LMDB_FILE, map_size=1 << 34)  # 16 GB
        with lmdb_index.begin(write=True) as txn:
            for embs, ctx in dataset:
                video_id = ctx["video_id"].numpy().decode()
                start_time = float(ctx["start_time_seconds"].numpy())
                labels = ctx["labels"].values.numpy().tolist()

                embs_np = embs.numpy()

                for j, emb in enumerate(embs_np):
                    buffer_embeddings.append(emb.astype(np.float32))
                    buffer_metadata.append(
                        (video_id, start_time + j, start_time + j + 1.0, labels))

                    # Flush in batches
                    if len(buffer_embeddings) >= batch_size:
                        index.add(np.stack(buffer_embeddings))
                        print("Adding row ", row_id)
                        for meta in buffer_metadata:
                            txn.put(str(row_id).encode(), pickle.dumps(meta))
                            row_id += 1
                        buffer_embeddings, buffer_metadata = [], []

            # Flush remainder
            if buffer_embeddings:
                index.add(np.stack(buffer_embeddings))
                for meta in buffer_metadata:
                    txn.put(str(row_id).encode(), pickle.dumps(meta))
                    row_id += 1

        faiss.write_index(index, AUDIOSET_INDEX_FILE)

    def sample_embeddings_for_training(self, dataset, num_samples=100000):
        sample = []
        n = 0
        # get spacing for random sampling of 20 million samples
        space = 20749183 // num_samples
        for embs, _ in dataset:
            embs_np = embs.numpy()
            for emb in embs_np:
                if n % space == 0:
                    print(f"Adding embedding {n} to training sample")
                    sample.append(emb.astype(np.float32))
                n += 1
        return np.stack(sample)

    # def _reservoir_sample_embeddings_for_training(self, dataset, num_samples=100000):
    #     """
    #     Reservoir sampling to collect training vectors from TFRecord dataset.
    #     """
    #     reservoir = []
    #     n = 0

    #     for embs, ctx in dataset:
    #         embs_np = embs.numpy()
    #         for emb in embs_np:
    #             if len(reservoir) < num_samples:
    #                 reservoir.append(emb.astype(np.float32))
    #             else:
    #                 j = random.randint(0, n)
    #                 if j < num_samples:
    #                     reservoir[j] = emb.astype(np.float32)
    #             n += 1

    #     return np.stack(reservoir)

    def _load_embeddings_and_metadata(self, dataset, limit=None):
        ds = dataset.take(limit)
        all_embeddings = []
        all_metadata = []
        for (embeddings, context) in dataset:
            video_id = context["video_id"].numpy().decode("utf-8")
            start_sec = float(context["start_time_seconds"].numpy())
            labels = context["labels"].values.numpy().tolist()

            embeddings_numpy = embeddings.numpy()

            for j, emb in enumerate(embeddings_numpy):
                all_embeddings.append(emb.astype(np.float32))
                all_metadata.append(
                    (video_id, start_sec + j, start_sec + j + 1.0, labels))

        return np.stack(all_embeddings), all_metadata


    def get_k_nearest_neighbors(self, embeddings, k=1, nprobe=16):
        self.index.nprobe = nprobe
        distances, indices = self.index.search(embeddings, k)
        results = []
        with self.lmdb.begin() as txn:
            for idx, dist in zip(indices[0], distances[0]):
                if idx == -1:
                    continue
                meta_bytes = txn.get(str(idx).encode())
                if meta_bytes is None:
                    continue
                meta = pickle.loads(meta_bytes)
                results.append((idx, dist, meta))

            for r in results:
                idx, dist, (video_id, start_time, end_time, labels) = r
                print(f"Index={idx}, Distance={dist:.4f}, Video=https://www.youtube.com/watch?v={video_id}&t={start_time:.0f}s, "
                      f"Segment={start_time:.1f}-{end_time:.1f}s, Labels={labels}")


def get_tfrecord_files(folder_path: str):
    # recursively get all the tfrecord files in the folder
    tfrecord_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.tfrecord'):
                tfrecord_files.append(os.path.join(root, file))
    return tfrecord_files


def main():
    print("Parsing TFRecords to embeddings")
    embedding_dataset = parse_tfrecords_to_embeddings(
        get_tfrecord_files('audioset_v1_embeddings'), True)

    print("Getting embeddings for the query audio")
    model = VGGishEmbeddingsModel()
    embeddings = model.get_embeddings('assets/YouDoSomethingToMe2.mp3', True)
    tf.keras.backend.clear_session()

    print("Loading FAISS index")
    faiss_loader = FaissLoader(embedding_dataset)
    print("Getting k nearest neighbors")
    faiss_loader.get_k_nearest_neighbors(embeddings, k=3)


if __name__ == "__main__":
    main()
