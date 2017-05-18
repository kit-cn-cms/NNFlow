import numpy as np
import sys

class DataFrame:

    def __init__(self, array, out_size, normalization):
        """Initializes the DataFrame.

        Arguments: 
        ----------------
        array (numpy ndarray):
            Array containing labels, data, and weights.
        out_size (int):
            Dimension of the output (i.e. label).
        normalization (string):
            String indicating whether to use Gaussian or min-max norm.
        """

        self.out_size = out_size
        self.normalization = normalization
        self.y = array[:, :self.out_size]
        self.x = array[:, self.out_size:-1]
        self.w = array[:, -1:]

        self.n = self.x.shape[0]
        self.nfeatures = self.x.shape[1]
        
        print('Found {} events.'.format(self.n))
        print('Length of each event: {}'.format(self.nfeatures))

        print('Now shuffling data...')
        self.shuffle()
        print('done.')
        # print(self.w[0:100])

    def normalize(self):
        """Normalizes the training data.
        """
        print('Now normalizing; this may take some minutes.')
        if (self.normalization == 'minmax'):
            print('Normalizing using Min-Max normalization.')
            x_max = np.amax(self.x, axis=0).astype(np.float32)
            x_min = np.amin(self.x, axis=0).astype(np.float32)
            self.x = 2.0*np.nan_to_num((self.x - x_min) / (x_max - x_min))-1.0


        elif (self.normalization == 'gaussian'):
            print('Normalizing using Gaussian normalization.')
            x_mean = np.mean(self.x, axis=0).astype(np.float32)
            x_std = np.std(self.x, axis=0).astype(np.float32)
            self.x = np.nan_to_num((self.x - x_mean) / x_std)

        else:
            sys.exit('Only minmax and gaussian normalization are available.')


    
    def shuffle(self):
        """Shuffles the data.
        """

        perm = np.random.permutation(self.n)
        self.x = self.x[perm]
        self.y = self.y[perm]
        self.w = self.w[perm]

        for i in range(self.n):
            if (self.x[i].shape[0] != self.nfeatures):
                print('Some data does not have the right shape.')
            if (self.y[i].shape[0] != self.out_size):
                print('Some labels do not have the right shape.')
            if (self.w[i].shape[0] != 1):
                print('Some weights do not have the right shape.')

        self.next_id = 0

    def next_batch(self, batch_size):
        """Returns the next batch of events.
        
        Arguments:
        ----------------
        batch_size (int):
            Size of each batch.
        """

        if (self.next_id + batch_size >= self.n):
            self.shuffle()

        cur_id = self.next_id
        self.next_id += batch_size

        return (np.array(self.x[cur_id:cur_id + batch_size]),
                np.array(self.y[cur_id:cur_id + batch_size]),
                np.array(self.w[cur_id:cur_id + batch_size]))

