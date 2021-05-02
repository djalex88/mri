"""
Some routines related to MRI reconstruction.

GRAPPA, SPIRiT, ESPIRiT, PRUNO

"""

import numpy as np


def grappa_kernel(f, R, r, v_size, num_nodes, acs_start, acs_stop, lamda=0):
    """
    Computes convolutional kernel for the GRAPPA reconstruction algorithm [1].
    
    Parameters
    ----------
    f : numpy array of shape (num_coils, size_y, size_x)
        k-space data with frequency encoding (readout direction)
        along Y-axis (axis=1) and phase encoding along X-axis (axis=2)
    R : int
        acceleration factor (2 <= R <= 8)
    r : int
        position of missing data between acquired data (1 <= r < R)
    v_size : int, odd
        vertical size of the kernel
    num_nodes : int, even
        number of acquired neighbour columns (e.g., 2, 4)
    acs_start : int
        index of first ACS data column
    acs_stop : int
        index of next to last ACS data column
    lamda : float, optional
        Tykhonov regularization
    
    Returns
    -------
    G : ndarray
        GRAPPA kernel
    
    References
    ----------
    [1]  M. A. Griswold, P. M. Jakob, R. M. Heidemann, M. Nittka,
    V. Jellus, J. Wang, B. Kiefer, and A. Haase,
    "Generalized autocalibrating partially parallel acquisitions (GRAPPA),"
    Magnetic Resonance in Medicine, vol. 47, no. 6, pp. 1202–1210, 2002.
    
    """
    
    assert 2 <= R <= 8
    assert r in range(1, R)
    assert v_size % 2
    assert num_nodes > 1 and num_nodes % 2 == 0
    
    num_coils, size_y, size_x = f.shape
    
    width = num_nodes + (num_nodes-1)*(R-1)
    
    assert acs_start + width <= acs_stop
    
    shift = r - (R+R%2)//2
    Ny = size_y - 2*(v_size//2)
    
    indices = [(slice(j-width//2, j+width//2+1-R%2, R), j+shift) for j in range(acs_start+width//2, acs_stop-width//2+R%2)]

    # fill calibration matrix
    A = np.empty(shape=(Ny*len(indices), num_coils*v_size*num_nodes), dtype=f.dtype)
    for j, (idx1, idx2) in enumerate(indices):
        for k in range(num_coils):
            for m in range(v_size):
                A[j*Ny:(j+1)*Ny, k*v_size*num_nodes+m*num_nodes:k*v_size*num_nodes+(m+1)*num_nodes] = f[k,m:m+Ny,idx1]
    
    Ah = np.conj(A.T)
    X = Ah @ A
    if lamda:
        X+= np.diag(np.full(X.shape[0], lamda))

    # select target data and solve
    b = f[:, v_size//2:v_size//2+Ny, [idx2 for idx1, idx2 in indices]].transpose(2,1,0).reshape(Ny*len(indices), num_coils)
    g = np.linalg.solve(X, Ah @ b).T.reshape(num_coils, num_coils, v_size, num_nodes)

    # create kernels
    G = np.zeros((num_coils, num_coils, v_size, width+abs(2*shift+R%2)), dtype=f.dtype)
    G[:,:,:,max(0,-2*shift-R%2)::R] = g[...]
        
    return np.ascontiguousarray(np.flip(G, axis=(-2,-1)))


def calibration_matrix(acs, width, height, mode='conv'):
    """
    Returns calibration matrix for the given ACS data and kernel size.
    
    Parameters
    ----------
    acs : numpy array of shape (num_coils, size_y, size_x)
        Auto-Calibration Signal
    width : int, odd
        width of the kernel
    height : int, odd
        height of the kernel
    mode : either 'conv' or 'corr'
        'conv' for convolution, 'corr' for correlation
    
    Returns
    -------
    A : ndarray
        calibration matrix
    
    """
    
    if mode == 'conv':
        acs = acs[:,::-1,::-1]
    else:
        assert mode == 'corr'
    
    num_coils, size_y, size_x = acs.shape
    Ny = size_y - 2*(height//2)
    kernel_size = width * height
    
    assert width % 2 and height % 2
    assert width <= size_x and height <= size_y
    
    indices = [slice(j-width//2, j+width//2+1) for j in range(width//2, size_x-width//2)]
    
    A = np.empty(shape=(Ny*len(indices), num_coils*kernel_size), dtype=acs.dtype)
    for j, idx1 in enumerate(indices):
        for k in range(num_coils):
            for m in range(height):
                A[j*Ny:(j+1)*Ny, k*kernel_size+m*width:k*kernel_size+(m+1)*width] = acs[k,m:m+Ny,idx1]
    
    return A


def spirit_kernel(f, kernel_width, kernel_height, acs_rect):
    """
    Computes SPIRiT kernel [1].
    
    Parameters
    ----------
    f : numpy array of shape (num_coils, size_y, size_x)
        k-space data
    kernel_width : int, odd
        width of the convolutional kernel
    kernel_height : int, odd
        height of the convolutional kernel
    acs_rect : tuple of two tuples of two integers, i.e., ((y1, y2), (x1, x2))
        describes ACS area
    
    Returns
    -------
    G : ndarray of shape (num_coils, num_coils, kernel_height, kernel_width)
        SPIRiT kernel
    
    References
    ----------
    [1]  M. Lustig and J. M. Pauly, "SPIRiT: Iterative self-consistent
    parallel imaging reconstruction from arbitrary k-space,"
    Magnetic Resonance in Medicine, vol. 64, no. 2, pp. 457–471, 2010.
    
    """
    
    (y1, y2), (x1, x2) = acs_rect
    
    A = calibration_matrix(f[:,slice(y1, y2),slice(x1, x2)], kernel_width, kernel_height)
    
    num_coils = f.shape[0]
    kernel_size = kernel_width * kernel_height
    
    target_idx = range(kernel_size//2, A.shape[1], kernel_size)
    
    Ah = np.conj(A.T)
    
    X = Ah @ A
    b = Ah @ A[:,target_idx]
    
    G = np.zeros((num_coils, num_coils*kernel_size), dtype=f.dtype)

    for i, k in enumerate(target_idx):
        idx1 = np.array([*range(k)] + [*range(k+1, X.shape[0])]).reshape(-1, 1)
        idx2 = idx1.transpose()
        g = np.linalg.solve(X[idx1,idx2], b[idx1.reshape(-1), i])
        G[i,:k], G[i,k+1:] = g[:k], g[k:]
        
    G.shape = num_coils, num_coils, kernel_height, kernel_width
    
    return G


def null_space(f, N, kernel_width, kernel_height, acs_rect):
    """
    Computes a set of kernels for null space operator (see PRUNO [1]).
    
    Parameters
    ----------
    f : numpy array of shape (num_coils, size_y, size_x)
        k-space data
    kernel_width : int, odd
        width of the convolutional kernel
    kernel_height : int, odd
        height of the convolutional kernel
    acs_rect : tuple of two tuples of two integers, i.e., ((y1, y2), (x1, x2))
        describes ACS area
    
    Returns
    -------
    G : ndarray of shape (N, num_coils, kernel_width, kernel_height)
        approximation of null space of calibration matrix
    
    References
    ----------
    [1]  J. Zhang, C. Liu, and M. E. Moseley,
    "Parallel reconstruction using null operations,"
    Magnetic Resonance in Medicine, vol. 66, no. 5, pp. 1241–1253, 2011.
    
    """
    
    (y1, y2), (x1, x2) = acs_rect
    
    A = calibration_matrix(f[:,slice(y1, y2),slice(x1, x2)], kernel_width, kernel_height)
    
    Vh = np.linalg.svd(A)[-1]
    G = Vh[-N:,:].conj().reshape(N, -1, kernel_width, kernel_height)
    
    return G


def espirit(f, kernel_width, kernel_height, acs_rect, num_basis=100, threshold=0.01):
    """
    A simple implementation of the ESPIRiT algorithm [1].
    
    Parameters
    ----------
    f : numpy array of shape (num_coils, size_y, size_x)
        k-space data
    kernel_width : int, odd
        width of the convolutional kernel
    kernel_height : int, odd
        height of the convolutional kernel
    acs_rect : tuple of two tuples of two integers, i.e., ((y1, y2), (x1, x2))
        describes ACS area
    num_basis : int
        number of basis vectors to use
    threshold : float
        threshold
    
    Returns
    -------
    s : ndarray of shape (num_coils, size_y, size_x)
        coil sensitivity estimates
    
    References
    ----------
    [1]  M. Uecker, P. Lai, M. J. Murphy, P. Virtue, M. Elad, J. M. Pauly,
    S. S. Vasanawala, and M. Lustig,
    "ESPIRiT – an eigenvalue approach to autocalibrating parallel MRI:
    Where SENSE meets GRAPPA,"
    Magnetic Resonance in Medicine, vol. 71, no. 3, pp. 990–1001, 2014.
    
    """
    
    def compute_GtG(g):
        N, C, H, W = g.shape
        h = np.zeros(shape=(C, C, 2*H-1, 2*W-1), dtype=g.dtype)
        for i in range(C):
            for j in range(C):
                for k in range(N):
                    h[i,j,...]+= np.convolve(np.flip(g[k,i,...].conj(), axis=(-2,-1)), g[k,j,...], mode='full')
        return h
    
    (y1, y2), (x1, x2) = acs_rect
    
    num_coils, size_y, size_x = f.shape
    
    A = calibration_matrix(f[:,slice(y1, y2),slice(x1, x2)], kernel_width, kernel_height)
    
    Vh = np.linalg.svd(A)[-1]
    
    # get basis vectors
    G = Vh[:num_basis].conj().reshape(-1, num_coils, kernel_height, kernel_width)
    G = compute_GtG(G)
    
    # do IFFT
    ky, kx = G.shape[-2:]
    pos_x = size_x//2 - kx//2
    pos_y = size_y//2 - ky//2
    g = np.zeros((num_coils, num_coils, size_y, size_x), dtype=G.dtype)
    g[..., pos_y:pos_y+ky, pos_x:pos_x+kx] = G[...]
    g = np.fft.ifft2(np.fft.ifftshift(g, axes=(-2,-1)), norm='ortho') * (size_x*size_y)**0.5
    g = np.ascontiguousarray(g.transpose(2,3,0,1).reshape(-1, num_coils, num_coils))
    
    w, s = np.linalg.eigh(g)
    
    # normalize eigvals with kernel size
    w = w.real / (kernel_width*kernel_height)
    
    # select eigenpairs corresponding to 1
    idx = w.argmax(axis=-1)
    w = w[np.arange(idx.size), idx]
    s = s[np.arange(idx.size), :, idx]
    
    s[w < (1-threshold)] = 0
    
    return np.ascontiguousarray(s.T.reshape(num_coils, size_y, size_x))

