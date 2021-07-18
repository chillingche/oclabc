import numpy as np
from numpy.core.fromnumeric import size
from numpy.matrixlib.defmatrix import _from_string
import torch
import torch.nn.functional as F


def dumpNpArr(nparr, txt):
    with open(txt, 'w') as f:
        for val in nparr:
            f.write("{}\n".format(val))

def printFlatten(nparr):
    num_elem = 1024 if nparr.size > 1024 else nparr.size
    flatarr = nparr.flatten()
    for i in range(0, num_elem):
        print(flatarr[i], end=' ')
        if ((i + 1) % 8 == 0):
            print("")

def get_indices(X_shape, HF, WF, stride, pad):
    """
        Returns index matrices in order to transform our input image into a matrix.

        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.

        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d. 
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    # get input size
    m, n_C, n_H, n_W = X_shape

    # get output size
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1
  
    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(HF), WF)
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----
    
    # Slide 1 vector.
    slide1 = np.tile(np.arange(WF), HF)
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d

def im2col(X, HF, WF, stride, pad):
    """
        Transforms our input image into a matrix.

        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.

        Returns:
        -cols: output matrix.
    """
    # Padding
    X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    # Multi-dimensional arrays indexing.
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols

def col2im(dX_col, X_shape, HF, WF, stride, pad):
    """
        Transform our matrix back to the input image.

        Parameters:
        - dX_col: matrix with error.
        - X_shape: input image shape.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.

        Returns:
        -x_padded: input image with error.
    """
    # Get input size
    N, D, H, W = X_shape
    # Add padding if needed.
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, H_padded, W_padded))

    # Index matrices, necessary to transform our input image into a matrix.
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    # Retrieve batch dimension by spliting dX_col N times: (X, Y) => (N, X, Y)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    # Reshape our matrix back to image.
    # slice(None) is used to produce the [::] effect which means "for every elements".
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    # Remove padding from new image if needed.
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[pad:-pad, pad:-pad, :, :]

def gemm(ic, ih, iw, oc, oh, ow, M, N, K, itensor, wtensor, otensor):
    for idx in range(0, (N + 3) // 4):
        for idy in range(0, (M + 3) // 4):
            in_idx = idx * 4
            w_idx = idy * 4
            cols_val = np.zeros((4, 4), dtype=np.float32)
            for ki in range(0, K):
                w4 = wtensor[ki * M + w_idx : ki * M + w_idx + 4]
                i4 = itensor[ki * N + in_idx : ki * N + in_idx + 4]
                cols_val[0] += w4[0] * i4
                cols_val[1] += w4[1] * i4
                cols_val[2] += w4[2] * i4
                cols_val[3] += w4[3] * i4

            im_val = np.zeros((2, 8), dtype=np.float32)
            im_val[0][0] = cols_val[0][0]
            im_val[0][1] = cols_val[1][0]
            im_val[0][2] = cols_val[0][1]
            im_val[0][3] = cols_val[1][1]
            im_val[0][4] = cols_val[0][2]
            im_val[0][5] = cols_val[1][2]
            im_val[0][6] = cols_val[0][3]
            im_val[0][7] = cols_val[1][3]

            im_val[1][0] = cols_val[2][0]
            im_val[1][1] = cols_val[3][0]
            im_val[1][2] = cols_val[2][1]
            im_val[1][3] = cols_val[3][1]
            im_val[1][4] = cols_val[2][2]
            im_val[1][5] = cols_val[3][2]
            im_val[1][6] = cols_val[2][3]
            im_val[1][7] = cols_val[3][3]

            oc_idx = w_idx // 4
            ih_idx = in_idx // iw
            iw_idx = in_idx % iw
            oh_idx = ih_idx * 2
            ow_idx = iw_idx * 2
            # if in_idx == 0 and w_idx == 0:
            otensor[oc_idx * oh * ow + oh_idx * ow + ow_idx : oc_idx * oh * ow + oh_idx * ow + ow_idx + 8] = im_val[0]
            otensor[oc_idx * oh * ow + oh_idx * ow + ow_idx + ow : oc_idx * oh * ow + oh_idx * ow + ow_idx + ow + 8] = im_val[1]
            # otensor[w_idx * N + in_idx : w_idx * N + in_idx + 4] = cols_val[0]
            # otensor[(w_idx + 1) * N + in_idx : (w_idx + 1) * N + in_idx + 4] = cols_val[1]
            # otensor[(w_idx + 2) * N + in_idx : (w_idx + 2) * N + in_idx + 4] = cols_val[2]
            # otensor[(w_idx + 3) * N + in_idx : (w_idx + 3) * N + in_idx + 4] = cols_val[3]



if __name__ == "__main__":
    np.random.seed(1313)
    # M: 128 * 2 * 2    K: 128    N: 3600
    ic = 8
    ih = 30
    iw = 30
    oc = 8
    oh = 60
    ow = 60
    M = oc * 2 * 2
    K = ic
    N = ih * iw
    np_in = np.random.randn(ic * ih * iw)
    np_in = np.resize(np_in, (1, ic, ih, iw))
    np_in = np.asarray(np_in, dtype=np.float32)
    in_tensor = torch.from_numpy(np_in)
    # print(in_tensor)

    np_weight = np.random.rand(ic * oc * 2 * 2)
    np_weight = np.resize(np_weight, (ic, oc, 2, 2))
    np_weight = np.asarray(np_weight, dtype=np.float32)
    weight = torch.from_numpy(np_weight)
    # print(weight)

    out_tensor = F.conv_transpose2d(in_tensor, weight, stride=2)

    # weightT = weight.reshape(ic, oc * 2 * 2)
    # weightT = weightT.permute(1, 0)

    # in_tensor = in_tensor.reshape(ic, ih * iw)
    # tmp = torch.matmul(weightT, in_tensor)

    # tmp = tmp.numpy()
    # gemm_out = col2im(tmp, (1, 4, 120, 120), 2, 2, 2, 0)
    # print(gemm_out, gemm_out.shape)

    in_tensor = in_tensor.numpy().flatten()
    weight = weight.numpy().flatten()
    dumpNpArr(in_tensor, "input.txt")
    dumpNpArr(weight, "weight.txt")


    # otensor = np.zeros(oc * oh * ow, dtype=np.float32)
    # gemm(ic, ih, iw, oc, oh, ow, M, N, K, in_tensor, weight, otensor)
    # tmp = tmp.numpy().flatten()
    # dumpNpArr(tmp, "gemm.txt")
    # print(tmp, tmp.shape)

    print(out_tensor, out_tensor.shape)
    dumpNpArr(out_tensor.numpy().flatten(), "output.txt")
    printFlatten(out_tensor.numpy())
    # print(np.sum(tmp != otensor))
