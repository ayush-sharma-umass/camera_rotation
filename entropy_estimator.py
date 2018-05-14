import numpy as np
import fast_histogram as fht

import time


class EntropyEstimator:

    def __init__(self):
        self.num_imgs = None
        self.hist = None
        self.bin_map = None

        pass


    def build_histogram(self, val_range, nbins, imgs):
        """

        :param val_range: tupe (start, end) excluding end
        :param nbins:
        :param imgs: numpy array (n, ht, wt)
        :return:
        """

        t1 = time.time()

        # converting RGB to grayscale assuming 4th axis is channel
        if len(imgs.shape) > 3:
            imgs = np.mean(imgs, axis=3)

        self.val_range = val_range
        self.nbins = nbins
        bin_sz = (val_range[1] - val_range[0])/float(nbins)

        # Making bin map

        self.bin_map = np.zeros(val_range[1] - val_range[0])
        for i in range(val_range[0], val_range[1]):
            self.bin_map[i] = int(i/bin_sz)



        # Creating histogram for eaxh spatial pixel position flattened out

        N, Ht, Wt = imgs.shape
        self.num_imgs = N
        imgs = imgs.transpose(1,2,0)
        imgs = imgs.reshape(Ht*Wt, -1)


        # initializing histogram and adding filler to avoid divide by zero error
        self.hist = np.zeros((Ht*Wt, nbins))
        filler = np.ones(nbins) *1.0

        for k in range(0, Ht*Wt):
            self.hist[k, :] = (fht.histogram1d(imgs[k,:], range=val_range, bins=nbins) + filler)

        print "time taken in creating histogram: ", time.time() - t1

        return self.hist

    def set_params(self, num_imgs, hist, nbins, val_range):
        self.num_imgs = num_imgs
        self.hist = hist
        self.val_range = val_range
        self.nbins = nbins
        self.bin_sz = (val_range[1] - val_range[0]) / float(nbins)



    def compute_entropy(self):
        return -np.sum(np.sum(np.log(self.hist/(self.num_imgs + self.nbins)), axis=1))

    def get_updated_entropy(self, old_img, new_img):
        """

        :param old_img:
        :param new_img:
        :return:
        """
        ht, wt = (0,0)
        if len(new_img.shape) > 2:
            ht, wt, _ = old_img.shape
            old_img = np.mean(old_img, axis=2)
            new_img = np.mean(new_img, axis=2)
        else:
            ht, wt = old_img.shape



        old_img = old_img.reshape(ht*wt, -1)
        old_bin_id = self.find_bin(old_img)
        old_bin_id = old_bin_id.astype(int)


        new_img = new_img.reshape(ht * wt, -1)
        new_bin_id = self.find_bin(new_img)
        new_bin_id = new_bin_id.astype(int)

        # print "old bin id: ", old_bin_id
        # print "new bin id: ", new_bin_id


        # X = np.copy(self.hist)

        self.hist[np.arange(0, ht * wt), old_bin_id] -= 1.0
        self.hist[np.arange(0, ht*wt), new_bin_id] += 1.0


        # print "HISTOGRAM update: ", self.hist

        # for i in range(0, ht*wt):
        #     if np.sum(self.hist[i, :] == 0) > 0:
        #         print i, " ", X[i,:], " " , self.hist[i, :], " ", old_img[i], " ", new_img[i]
        #
        # print "AF Is = zero: ", np.sum(self.hist ==0.0)


        ent = self.compute_entropy()

        self.hist[np.arange(0, ht * wt), old_bin_id] += 1.0
        self.hist[np.arange(0, ht * wt), new_bin_id] -= 1.0

        return ent


    def find_bin(self, value):
        """

        :param value: a double or a numpy array
        :return: a double or numpy array of the id of the value
        """

        binid = value/self.bin_sz
        return np.squeeze(binid.astype(int), axis=1)




# x = np.array([0.,0.,3.,1.,4.,2.,5.])
# x = x.astype(int)
# a = np.zeros((7,6))
# b = np.arange(0,7)
#
# a[b,x] += 2
#
# print a

#
# x = np.array([1,2,3.33, 4,5,2,6.66,1,8,9])
# y, e = np.histogram(x, bins=3, range=(0,10))
# bin_sz = 10.0/3
# print y, e
#
# z = x/bin_sz
# z = z.astype(int)
# print z
#
# print np.random.randint(0, 10, (3,3))