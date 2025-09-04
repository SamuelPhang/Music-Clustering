import tensorflow as tf
import numpy as np
import os


def parse_tfrecords_to_embeddings(tfrecord_files, dequantized=True):
    """
    Parses TFRecord files to extract audio embeddings.

    Args:
        tfrecord_files (list or str): A single path or a list of paths to TFRecord files.

    Returns:
        list: A list of NumPy arrays, where each array is a 128-dimensional embedding.
    """
    if not isinstance(tfrecord_files, list):
        tfrecord_files = [tfrecord_files]

    # 1. Create a feature description to parse the TFRecord.
    # This acts as a schema to tell TensorFlow what to look for.
    feature_description = {
        "video_id": tf.io.FixedLenFeature([], tf.string),
        "start_time_seconds": tf.io.FixedLenFeature([], tf.float32),
        "end_time_seconds": tf.io.FixedLenFeature([], tf.float32),
        "labels": tf.io.VarLenFeature(tf.int64),
    }

    sequence_features = {
        "audio_embedding": tf.io.FixedLenSequenceFeature([], tf.string)
    }

    # 2. Define a parsing function.
    def _parse_function(example_proto):
        # Parse the input `example_proto` using the dictionary above.
        # return tf.io.parse_single_example(example_proto, feature_description)
        context, sequence = tf.io.parse_single_sequence_example(
            example_proto, feature_description, sequence_features)
        embeddings = tf.map_fn(
            lambda x: tf.io.decode_raw(x, tf.uint8),
            sequence["audio_embedding"],
            fn_output_signature=tf.uint8
        )

        if dequantized:
            embeddings = tf.cast(embeddings, tf.float32)
            embeddings = (embeddings - 128.0) / 128.0

        return embeddings, context

    # 3. Create a dataset and map the parsing function.
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset
    # embeddings_list = []
    # # 4. Iterate through the parsed dataset to extract the embeddings.
    # for features in parsed_dataset:
    #     # Get the raw audio_embedding string.
    #     raw_embedding = features['audio_embedding'].numpy()

    #     # Decode the raw bytes into a NumPy array of uint8.
    #     # The AudioSet embeddings are stored as 128 quantized 8-bit integers.
    #     embedding = np.frombuffer(raw_embedding, dtype=np.uint8)

    #     # De-quantize the embedding back to floating-point values (0.0 to 1.0).
    #     embedding = embedding / 255.0

    #     embeddings_list.append(embedding)

    # return embeddings_list


def main():
    # --- Example Usage ---
    # Assume you have a TFRecord file named 'audioset_vggish.tfrecord'
    # # Create a dummy file for this example:
    # with tf.io.TFRecordWriter("audioset_v1_embeddings/bal_train/__.tfrecord") as writer:
    #     for i in range(5):
    #         embedding_data = np.random.randint(0, 256, size=128, dtype=np.uint8)
    #         example = tf.train.Example(features=tf.train.Features(feature={
    #             'video_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[f'vid_{i}'.encode('utf-8')])),
    #             'start_time_seconds': tf.train.Feature(float_list=tf.train.FloatList(value=[float(i)])),
    #             'end_time_seconds': tf.train.Feature(float_list=tf.train.FloatList(value=[float(i+10)])),
    #             'audio_embedding': tf.train.Feature(bytes_list=tf.train.BytesList(value=[embedding_data.tobytes()]))
    #         }))
    #         writer.write(example.SerializeToString())

    # Now, parse the file you created or downloaded
    my_embeddings = parse_tfrecords_to_embeddings(
        'audioset_v1_embeddings/bal_train/__.tfrecord', True)

    count = 0
    for embeddings, context in my_embeddings:
        video_id = context["video_id"].numpy().decode("utf-8")
        print(f"\nVideo ID: {video_id}")
        print("Embeddings shape:", embeddings.shape)

        # Print first few embedding vectors (per-second)
        for i, vec in enumerate(embeddings.numpy()[:5]):  # first 5 timesteps
            print(f"Step {i} embedding (first 10 dims):", vec[:10])

        count += 1
        if count >= 1:  # stop after first example
            break


if __name__ == "__main__":
    main()
