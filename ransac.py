import numpy as np
import random

# main function for estimation of rigid body transform
def findRigidBody(pts1, pts2, minPoints, numIter, thresh):
    assert pts1.shape[1] == 3 and pts2.shape[1] == 3
    estFunc = estimateRigidTransform
    distFunc = calcDistPointPoint
    
    TF_ref, maxInlrRef, mask = ransac(pts1, pts2, minPoints, numIter, thresh, estFunc, distFunc)    
    
    return TF_ref, mask

# ransac as function, flexible (what it's estimating depends on estFunc and distFunc)
def ransac(pts1, pts2, minPoints, numIter, thresh, estFunc, distFunc):

    N = pts1.shape[0]
    maxInlr = 0
    TF_ref = None
    maxInlrRef = None
    mask = np.zeros(N, dtype=np.int32)
    for i in range(numIter):
        # sample minimum number of points from pts1
        idx = np.asarray(random.sample(range(N), minPoints))

        ptsSample1 = pts1[idx, :]
        ptsSample2 = pts2[idx, :]

        # estimate function
        TF = estFunc(ptsSample1, ptsSample2)

        # get distances to function of every point in pts2
        d = distFunc(pts1, pts2, TF)
        
        # get number of inliers
        numInlr = np.sum(d < thresh)

        if numInlr > maxInlr:
            maxInlr = numInlr

            # refine by using all inliers
            idx_ref = d < thresh
            if np.sum(idx_ref) > minPoints:
                pts1_ref = pts1[idx_ref, :]
                pts2_ref = pts2[idx_ref, :]
                TF_ref = estFunc(pts1_ref, pts2_ref)

                dRefined = distFunc(pts1, pts2, TF_ref)

                maxInlrRef = np.sum(dRefined < thresh)
                
                mask = (dRefined < thresh).astype(int)
            
    return TF_ref, maxInlrRef, mask

# distance calculation. calculates the distances between two sets of points 
# assumes that there are correspondences and that r.g. the first point in pts1 corresponds to the first point in pts2
def calcDistPointPoint(pts1, pts2, TF):

    # homogeneous coordinates
    pts2_hom = np.append(pts2, np.ones((pts2.shape[0], 1)), axis=1)
    
    # transform
    pts2_TF = np.dot(pts2_hom, TF)
    
    # remove "1" from homoogeneous coordinates
    pts2_TF = pts2_TF[:, 0:3]
    
    # get vector of distances (length equal to number of points)
    d = np.linalg.norm(pts1-pts2_TF, ord=2, axis=1)

    return d

# estimate rigid body transformation from two sets of point corespondeces. At least three point pairs required. 
def estimateRigidTransform(pts1, pts2):
    N = pts1.shape[0]
    # we need at least 3 points
    if N < 3:
        print("Not enough points to estimate transform. At least 3 points needed")
        
    # keep going with the real algorithm, follow the paper
    d = pts1.T
    m = pts2.T

    cd = np.mean(d, axis=1)
    cm = np.mean(m, axis=1)

    # centered (set centroid to 0)
    d_c = (d.T - cd).T
        
    m_c = (m.T - cm).T

    H = np.dot(m_c, d_c.T)
    U, S, V = np.linalg.svd(H) # such that H = U * S * V, and NOT H = u * S * V.T

    R = np.dot(V.T, U.T)
    t = cd - np.dot(R, cm)

    # build the transform matrix, such that [d 1] = TF * [m 1]
    TF = np.eye(4)
    TF[0:3, 0:3] = R
    TF[0:3, 3] = t
    TF = TF.T
    return TF
    
# distance from a set of points to a plane defined by (normal, point)
def distToPlane(pts, pts2, plane):
    n, q = plane
    pts_q = pts-q
    d = np.dot(pts_q, n)
    return np.fabs(d)

# least squares estimation of a plane given three or more points in 3D
def estimatePlane(pts, pts1):
    N, dim = pts.shape
    assert dim == 3
    
    A = np.concatenate((pts[:, :2], np.ones((N, 1))), axis=1)
    b = pts[:, 2]
    res = np.linalg.lstsq(A, b)

    # res[0] is the normal vector, so
    n = np.concatenate((res[0][:2], [-1]), axis=0)
    n = n / np.linalg.norm(n)

    # for q0: get a point on the plane, at x = y = 1
    z = np.dot(res[0], np.asarray([1, 1, 1]))
    q = np.asarray([1, 1, z])
    
    # return normal vector and point on plane
    plane = (n, q)
    return plane

    # return the transposed matrix, such that [pts1 1] * TF = [pts2 1]
    return TF.T