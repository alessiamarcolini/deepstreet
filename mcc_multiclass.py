import numpy as np

def multimcc(t,p, classes=None):
    """ Matthews Correlation Coefficient for multiclass
    :Parameters:
        t : 1d array_like object integer
          target values
        p : 1d array_like object integer
          predicted values
        classes: 1d array_like object integer containing
          all possible classes

    :Returns:
        MCC : float, in range [-1.0, 1.0]
    """

    # Cast to integer
    tarr = np.asarray(t, dtype=np.int)
    parr = np.asarray(p, dtype=np.int)


    # Get the classes
    if classes is None:
        classes = np.unique(tarr)

    nt = tarr.shape[0]
    nc = classes.shape[0]

    # Check dimension of the two array
    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")

    # Initialize X and Y matrices
    X = np.zeros((nt, nc))
    Y = np.zeros((nt, nc))

    # Fill the matrices
    for i,c in enumerate(classes):
        yidx = np.where(tarr==c)
        xidx = np.where(parr==c)

        X[xidx,i] = 1
        Y[yidx,i] = 1

    # Compute the denominator
    denom = cov(X,X) * cov(Y,Y)
    denom = np.sqrt(denom)

    if denom == 0:
        # If all samples assigned to one class return 0
        return 0
    else:
        num = cov(X,Y)
        return num / denom


def confusion_matrix(t, p):
    """ Compute the multiclass confusion matrix
    :Parameters:
        t : 1d array_like object integer (-1/+1)
          target values
        p : 1d array_like object integer (-1/+1)
          predicted values

    :Returns:
        MCC : float, in range [-1.0, 1.0]
    """

    # Read true and predicted classes
    tarr = np.asarray(t, dtype=np.int)
    parr = np.asarray(p, dtype=np.int)

    # Get the classes
    classes = np.unique(tarr)

    # Get dimension of the arrays
    nt = tarr.shape[0]
    nc = classes.shape[0]

    # Check dimensions should match between true and predicted
    if tarr.shape[0] != parr.shape[0]:
        raise ValueError("t, p: shape mismatch")

    # Initialize Confusion Matrix C
    C = np.zeros((nc, nc))

    # Fill the confusion matrix
    for i in range(nt):
        ct = np.where(classes == tarr[i])[0]
        cp = np.where(classes == parr[i])[0]
        C[ct, cp] += 1

    # return the Confusion matrix and the classes
    return C, classes

def cov(x,y):
    nt = x.shape[0]

    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xxm = x - xm
    yym = y - ym

    tmp = np.sum(xxm * yym, axis=1)
    ss = tmp.sum()

    return ss/nt


if __name__ == '__main__':

    import numpy as np
    ytrue = np.array([ 7.,  6.,  2.,  2.,  2.,  2.,  6.,  2.,  2.,  3.,
                       2.,  7.,  6., 2.,  5.,  7.,  2.,  2.,  2.])

    ypred = np.array([7,  3,  6,  2,  2,  2,  3,  6,  2,
                      3,  2,  5,  3,  2,  2,  3,  2, 2,  2])

    ypred = np.copy(ytrue)
    ypred[0] = 6.0

    # ytrue = np.array([9., 2., 1., 2., 1., 5., 3., 9.,10., 4., 8., 5., 5., 3., 2., 9., 4., 3., 7., 5., 7., 9.,10., 7., 4., 1., 4., 1., 6., 6.,1., 9., 4., 4., 3., 6., 5., 3., 1., 1., 7.,10., 6., 1., 8., 1., 9., 1., 5., 4., 2., 1., 2., 8., 2., 1., 3., 7., 5., 5.,1., 4.,10., 8., 5., 7., 6., 2., 1., 3., 4.,10., 2., 4., 6.,6., 8.,10., 1., 5., 4., 2., 8., 1., 7., 1.,10., 9., 3., 1.,2., 3., 1., 2., 6., 3., 1., 6., 3., 3., 6., 1., 1., 2., 1.,2., 6., 2., 1., 8., 6., 5., 7., 1.,10., 4.,10., 8., 7.,10.,8., 3., 2., 9., 3., 3., 2., 2., 8., 7., 5., 9., 6., 2., 2.,4., 1., 2., 1., 5., 4., 1., 4., 4., 1., 4., 4., 1., 2., 5.,2., 1., 6., 2., 1., 9., 4., 1., 2., 2., 6., 4., 1., 2., 4.,1., 1., 4., 4., 2., 2., 2., 8., 3., 1., 1., 8., 2., 1., 6.,7., 1., 2., 8.,10., 1., 2., 4., 4.,10., 7., 1., 5., 7., 8.,5., 4., 1., 1., 2.,10., 2.])

    # ypred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    print(multimcc(ytrue, ypred))
