from keras.datasets import imdb
import numpy as np



class KIMDB_Data_Utils():

    def __init__(self):
        return

    def fetch_imdb_data(self, num_words=10000):
        """
        :param num_words: This arguments means that we want to keep the top 10,000 most frequently occuring words in the training data. Rare words will be discarded
        :return: The variables train_data and test_data are lists of reviews, each review being a list of word indices (encoding a sequence of words).  train_labels and test_labels are lists of 0s and 1s, where 0 stands for "negative" \
        and 1 stands for "positive":
        """
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)

        return (train_data, train_labels), (test_data, test_labels)

    def decode_review(self, train_data, index=0):
        """
        Return a decoded review
        :param index: is index into mapping of words into the integer index
        :return: a string matching the review
        """
        # word_index is a dictionary mapping words to an integer index
        word_index = imdb.get_word_index()
        # We reverse it, mapping integer indices to words
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        # We decode the review; note that our indices were offset by 3
        # because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
        decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[index]])

        return decoded_review

    def prepare_vectorized_sequences(self, sequences, dimension=10000):
        
        # Create an all-zero matrix of shape (len(sequences), dimension)
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.  # set specific indices of results[i] to 1s

        return results

    def prepare_vectorized_labels(self, labels):
        
        return np.asarray(labels).astype('float32')
#
# Test the functions
#
if __name__ == '__main__':
    # create a class handle
    kdata_cls = KIMDB_Data_Utils()
    (train_data, train_labels), (test_data, test_labels) = kdata_cls.fetch_imdb_data(num_words=10000)
    print(train_data[0])
    print(len(train_data))
    decoded = kdata_cls.decode_review(train_data)
    print(decoded)
    x_train = kdata_cls.prepare_vectorized_sequences(train_data)
    x_test = kdata_cls.prepare_vectorized_sequences(test_data)
    print(x_train[0])
    print(x_test[0])
    y_train = kdata_cls.prepare_vectorized_labels(train_labels)
    y_test = kdata_cls.prepare_vectorized_labels(test_labels)
    print(y_train)
    print(y_test)
