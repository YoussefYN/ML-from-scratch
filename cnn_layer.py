# This is a Bonus Question 
# It is about Same Convolution

# There are two functions you need to implement, Generic implementation, From Scratch:
#      (a) zero_padding
#      (b) convolution

# Don't do any additional imports, everything is already there

import numpy as np


#TO DO ---- 4 POINTS ---------- Implement the fit function ---------------------
def zero_padding(X, padding_size):
    """
    Pad with zeros all images of the dataset X
    The padding should be applied to the height and width only

    Inputs:
    X - numpy array of shape (batch_size, height, width, channels)
    padding_size - amount of padding around each image on vertical
        and horizontal dimensions

    Output:
    X_padding - padded image of shape
        (batch_size, height + 2*padding_size, width + 2*padding_size, channels)
    """

    ###################     Your code here     ########################
    def pad_one_image(img):
        col_zeros = np.zeros(shape=(img.shape[0], padding_size))
        row_zeros = np.zeros(shape=(padding_size, img.shape[1] + 2 * padding_size))
        img = np.append(img, col_zeros, axis=1)
        img = np.append(col_zeros, img, axis=1)
        img = np.append(img, row_zeros, axis=0)
        img = np.append(row_zeros, img, axis=0)
        return img
    return np.array([pad_one_image(x_i) for x_i in X])


#TO DO ---- 4 POINTS ---------- Implement the Convolution function ---------------------
def convolution(X, kernel):
    """
    Function should convolve input matrix with kernel

    Inputs:
    X - numpy array of shape (height, width)
    kernel - kernal of shape (kernel_height, kernel_width)

    Output:
    X_convolved - matrix after convolution of shape
        (height, width)
    """

    X = zero_padding(X.reshape(1, X.shape[0], X.shape[1]), int((kernel.shape[0] - 1) / 2))[0]
    dim1 = X.shape[0] - kernel.shape[0] + 1
    dim2 = X.shape[1] - kernel.shape[1] + 1
    ans = np.zeros(shape=(dim1, dim2))
    ###################     Your code here     ########################
    for i in range(dim1):
        for j in range(dim2):
            ans[i][j] = np.sum(np.multiply(kernel, X[i: i + kernel.shape[0], j: j + kernel.shape[1]]))
    return ans


def main():
    X = np.array([[2, 2, 3, 3, 3],
                  [0, 1, 3, 0, 3],
                  [2, 3, 0, 1, 3],
                  [3, 3, 2, 1, 2],
                  [3, 3, 0, 2, 3]])
    kernel = np.array([[2, 0, 1],
                       [1, 0, 0],
                       [0, 1, 1]])
    print(convolution(X, kernel))
    #    Result shold be:
    #    [[0.11111111 0.6666667  0.5555556  0.6666667  0.6666667 ]
    #     [0.7777778  1.1111112  1.         1.7777778  1.        ]
    #     [0.7777778  1.1111112  0.8888889  1.3333334  0.33333334]
    #     [1.         1.1111112  1.3333334  1.1111112  0.6666667 ]
    #     [0.33333334 1.2222222  1.1111112  0.6666667  0.44444445]]


if __name__ == '__main__':
    main()
