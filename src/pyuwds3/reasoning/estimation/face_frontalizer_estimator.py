import cv2
import numpy as np
import copy
from scipy import ndimage

WD = 250
HT = 250
ACC_CONST = 800


class FaceFrontalizerEstimator(object):
    """Python implementation of :
    Tal Hassner, Shai Harel*, Eran Paz* and Roee Enbar,
    Effective Face Frontalization in Unconstrained Images,
    IEEE Conf. on Computer Vision and Pattern Recognition (CVPR 2015)

    modified from https://github.com/ChrisYang/facefrontalisation"""
    def __init__(self, ref_3d_model_filename):
        self.points_3d = np.load(face_3d_model_filename)
        raise NotImplementedError()

    def estimate(self, rgb_image, face, camera_matrix, dist_coeffs):
        if "facial_landmarks" in face.features:
            points_2d = face.features["facial_landmarks"].data

            facebb = [face.xmin, face.ymin, face.width(), face.height()]
            w = facebb[2]
            h = facebb[3]
            fb_ = np.clip([[facebb[0] - w, facebb[1] - h], [facebb[0] + 2 * w, facebb[1] + 2 * h]], [0,0], [img_.shape[1], img_.shape[0]])
            img = rgb_image[fb_[0][1]:fb_[1][1], fb_[0][0]:fb_[1][0], :]
            p2d = copy.deepcopy(points_2d)
            p2d[:, 0] = (points_2d[:, 0] - fb_[0][0]) * float(WD) / float(img.shape[1])
            p2d[:, 1] = (points_2d[:, 1] - fb_[0][1])  * float(HT) / float(img.shape[0])
            img = cv2.resize(img, (WD, HT))
            tem3d = np.reshape(self.refU, (-1, 3), order='F')


            bgids = tem3d[:, 1] < 0# excluding background 3d points
            # plot3d(tem3d)
            # print tem3d.shape
            ref3dface = np.insert(tem3d, 3, np.ones(len(tem3d)),axis=1).T
            ProjM = self.get_headpose(p2d)[2]
            proj3d = ProjM.dot(ref3dface)
            proj3d[0] /= proj3d[2]
            proj3d[1] /= proj3d[2]
            proj2dtmp = proj3d[0:2]
            #The 3D reference is projected to the 2D region by the estimated pose
            #The check the projection lies in the image or not
            vlids = np.logical_and(np.logical_and(proj2dtmp[0] > 0, proj2dtmp[1] > 0),
                                   np.logical_and(proj2dtmp[0] < img.shape[1] - 1,  proj2dtmp[1] < img.shape[0] - 1))
            vlids = np.logical_and(vlids, bgids)
            proj2d_valid = proj2dtmp[:, vlids]

            sp_ = self.refU.shape[0:2]
            synth_front = np.zeros(sp_,np.float)
            inds = np.ravel_multi_index(np.round(proj2d_valid).astype(int),(img.shape[1], img.shape[0]),order = 'F')
            unqeles, unqinds, inverids, conts = np.unique(inds, return_index=True, return_inverse=True, return_counts=True)
            tmp_ = synth_front.flatten()
            tmp_[vlids] = conts[inverids].astype(np.float)
            synth_front = tmp_.reshape(synth_front.shape, order='F')
            synth_front = cv2.GaussianBlur(synth_front, (17,17), 30).astype(np.float)

            rawfrontal = np.zeros((self.refU.shape[0], self.refU.shape[1], 3))
            for k in range(3):
                z = img[:, :, k]
                intervalues = ndimage.map_coordinates(img[:, :, k].T, proj2d_valid,order=3, mode='nearest')
                tmp_ = rawfrontal[:, :, k].flatten()
                tmp_[vlids] = intervalues
                rawfrontal[:, :, k] = tmp_.reshape(self.refU.shape[0:2], order='F')

            mline = synth_front.shape[1]/2
            sumleft = np.sum(synth_front[:, 0:mline])
            sumright = np.sum(synth_front[:, mline:])
            sum_diff = sumleft - sumright
            print sum_diff
            if np.abs(sum_diff) > ACC_CONST:
                weights = np.zeros(sp_)
                if sum_diff > ACC_CONST:
                    weights[:, mline:] = 1.
                else:
                    weights[:, 0:mline] = 1.
                weights = cv2.GaussianBlur(weights, (33, 33), 60.5).astype(np.float)
                synth_front /= np.max(synth_front)
                weight_take_from_org = 1 / np.exp(1 + synth_front)
                weight_take_from_sym = 1 - weight_take_from_org
                weight_take_from_org = weight_take_from_org * np.fliplr(weights)
                weight_take_from_sym = weight_take_from_sym * np.fliplr(weights)
                weights = np.tile(weights, (1,3)).reshape((weights.shape[0],weights.shape[1],3),order='F')
                weight_take_from_org = np.tile(weight_take_from_org, (1,3)).reshape((weight_take_from_org.shape[0],weight_take_from_org.shape[1],3),order='F')
                weight_take_from_sym = np.tile(weight_take_from_sym, (1,3)).reshape((weight_take_from_sym.shape[0],weight_take_from_sym.shape[1],3),order='F')
                denominator = weights + weight_take_from_org + weight_take_from_sym
                frontal_sym = (rawfrontal * weights + rawfrontal * weight_take_from_org + np.fliplr(rawfrontal) * weight_take_from_sym) / denominator
            else:
                frontal_sym = rawfrontal
            return True, rawfrontal, frontal_sym
        return False, None, None
