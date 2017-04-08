class SVHN(object):

    path = ""

    def __init__(self, data_dir):
        """
            data_directory : path like /home/rajat/mlproj/dataset/
                            includes the dataset folder with '/'
            Initialize all your variables here
        """

    def train(self):
        """
            Trains the model on data given in path/train.csv

            No return expected
        """

    def get_sequence(self, image):
        """
            image : a variable resolution RGB image in the form of a numpy array

            return: list of integers with the sequence of digits. Example: [5,0,3] for an image having 503 as the sequence.

        """

    def save_model(self, **params):

        # file_name = params['name']
        # pickle.dump(self, gzip.open(file_name, 'wb'))

        """
            saves model on the disk

            no return expected
        """

    @staticmethod
    def load_model(**params):

        # file_name = params['name']
        # return pickle.load(gzip.open(file_name, 'rb'))

        """
            returns a pre-trained instance of SVHN class
        """

if __name__ == "__main__":
        # obj = SVHN('dataset/')
        # obj.train()
        # obj.save_model(name="svhn.gz")
