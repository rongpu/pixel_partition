from __future__ import division, print_function
import sys
import os
import time
import numpy as np
import healpy as hp


def find_survey_boundary(nside, ra, dec, min_neighbors=8):
    '''
    Find pixels that has fewer than specified number of neighbors (occupied
    pixels).

    Inputs
    ------
    nside: NSIDE parameter in healpix;
    ra, dec: coordinates of the healpix pixels;

    Output
    ------
    idx: indices of the pixels at survey boundary.
    '''

    # pixel indices in healpy
    pix_id = hp.pixelfunc.ang2pix(nside, ra, dec, lonlat=True)
    # count number of neighbors
    neighbors = hp.get_all_neighbours(nside, ra, dec, lonlat=True).T
    mask = np.in1d(neighbors, pix_id).reshape(neighbors.shape)
    neighbors_count = np.sum(mask, axis=1)

    idx = np.where(neighbors_count < min_neighbors)

    return idx


def _spread_label(index, neighbors, labels, label_now, rc_limit, rc=1):
    '''
    Recursively assign the label to all connected pixels;
    Return True if recusion finished, otherwise return False.
    '''

    if labels[index] == -1:
        labels[index] = label_now
    if rc >= rc_limit-100:
        return False

    recursion_flag = True # True if recursion finished
    spread_idx = np.where(neighbors[index] >= 0)[0]
    spread_idx = spread_idx[labels[neighbors[index][spread_idx]] == -1]
    for index1 in neighbors[index][spread_idx]:
        if labels[index1] == -1:
            recursion_flag = recursion_flag and _spread_label(index1, 
                neighbors, labels, label_now, rc_limit, rc=rc+1)
    
    return recursion_flag

def identify_islands(nside, ra, dec):
    '''
    Label the pixels by their island; (islands are pixels that are
    connected to each other); lone pixels are given the index -1.
    Note: might need to set a higher recursion limit for it to work,
    e.g.: sys.setrecursionlimit(3000)

    Inputs
    ------
    nside: NSIDE parameter in healpix;
    ra, dec: coordinates of the healpix pixels;

    Output
    ------
    labels: island index of each pixel.
    '''

    # Get max recursion depth
    rc_limit = sys.getrecursionlimit()

    ############## find neighbors ###############
    npix = hp.nside2npix(nside)

    # pixel indices in healpy
    pix_id = hp.pixelfunc.ang2pix(nside, ra, dec, lonlat=True)
    # find neighbors
    neighbors = hp.get_all_neighbours(nside, ra, dec, lonlat=True).T
    mask = np.in1d(neighbors, pix_id).reshape(neighbors.shape)
    # assign index number -1 to unoccupied neighboring pixels
    neighbors[~mask] = -1

    # Convert the neighbors array from healpix indices to generic
    # numpy indices
    pointer = -99 * np.ones(npix, dtype=int)
    for index in range(len(ra)):
        pointer[pix_id[index]] = index
    neighbors[mask] = pointer[neighbors[mask]]

    ########## label connected boundary pixels ############
    # Recursively assign island index;
    # It's messy because of the maximum recursion depth
    label_now = 0
    labels = -1 * np.ones(len(ra), dtype=int)
    while not np.all(labels != -1):
        index = np.argmax(labels == -1)
        labels[index] = label_now
        while True:
            recursion_flag = True
            idx = np.where(labels == label_now)[0]
            for index in idx:
                recursion_flag = recursion_flag and _spread_label(index, 
                    neighbors, labels, label_now, rc_limit)
            if recursion_flag:
                break
        label_now += 1

    return labels


def find_group_border(nside, ra, dec, labels):
    '''
    Find pixels that have neighbors that belong to a different group.
    Note that this does not include pixels at the survey boundary 
    unless the above criteria is true.

    Inputs
    ------
    nside: NSIDE parameter in healpix;
    ra, dec: coordinates of the healpix pixels;
    labels: group label of each pixel;

    Output
    ------
    idx: indices of the pixels at group borders.
    '''

    npix = hp.nside2npix(nside)

    # pixel indices in healpy
    pix_id = hp.pixelfunc.ang2pix(nside, ra, dec, lonlat=True)
    # find neighbors
    neighbors = hp.get_all_neighbours(nside, ra, dec, lonlat=True).T
    mask = np.in1d(neighbors, pix_id).reshape(neighbors.shape)
    # assign index number -1 to unoccupied neighboring pixels
    neighbors[~mask] = -1

    # Convert the neighbors array from healpix indices to generic
    # numpy indices
    pointer = -99 * np.ones(npix, dtype=int)
    for index in range(len(ra)):
        pointer[pix_id[index]] = index
    neighbors[mask] = pointer[neighbors[mask]]

    # Find bordering pixels
    mask1 = labels[neighbors] != labels[:, None]
    mask2 = neighbors >= 0
    mask_border = np.any(mask1 & mask2, axis=1)
    idx = np.arange(len(ra))[mask_border]

    return idx
