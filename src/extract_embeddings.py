import numpy as np

from essentia.standard import MonoLoader, TensorflowPredictVGGish
# pip install essentia-tensorflow
from scipy.spatial.distance import cosine


AUDIO_FILENAME = 'assets/YouDoSomethingToMe2.mp3'
AUDIO_FILENAME_2 = 'assets/AfterGlow.mp3'


class EmbeddingsModel:
    def get_embeddings(self, audio_filename: str, normalize=False) -> np.ndarray:
        audio = MonoLoader(filename=audio_filename,
                           sampleRate=16000, resampleQuality=4)()
        embeddings = self.model(audio)

        # Example normalization
        if normalize:
            embeddings = embeddings / \
                np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings


class VGGishEmbeddingsModel(EmbeddingsModel):
    def __init__(self):
        self.model = TensorflowPredictVGGish(
            graphFilename='model/audioset-vggish-3.pb', output="model/vggish/embeddings")


class DiscogsEmbeddingsModel(EmbeddingsModel):
    def __init__(self):
        # super().__init__(graph_filename='model/discogs-effnet-bs64-1.pb')
        pass


def main():
    vgg_model = VGGishEmbeddingsModel()
    embeddings1 = vgg_model.get_embeddings(AUDIO_FILENAME, True)
    embeddings2 = vgg_model.get_embeddings(AUDIO_FILENAME_2, True)
    # get the mean of the embeddings
    embeddings1 = np.mean(embeddings1, axis=0)
    embeddings2 = np.mean(embeddings2, axis=0)
    print(embeddings1)
    print(embeddings2)
    # get the cosine similarity
    print(1 - cosine(embeddings1, embeddings2))


if __name__ == "__main__":
    main()
