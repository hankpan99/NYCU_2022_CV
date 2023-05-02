import cv2
import numpy as np
import random
import math
import sys
import matplotlib.pyplot as plt
import argparse
import glob
from tqdm import tqdm, trange

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description='argument settings')
    parser.add_argument('-a', '--all', help='type = "bool", both baseline and bonus if True', action='store_true')
    parser.add_argument('-t', '--testcase', help='type = "str", specify the test case', type=str, default="baseline")

    args = parser.parse_args()

    return args

class ImageStiching():
    def __init__(self, args):
        baseline_pth = sorted(glob.glob('./baseline/*'))
        bonus_pth = sorted(glob.glob('./bonus/*'))

        if args.all:
            self.testcase_list = ['baseline', 'bonus']
            self.img_pth_list = [baseline_pth, bonus_pth]

        elif args.testcase == 'baseline':
            self.testcase_list = ['baseline']
            self.img_pth_list = [baseline_pth]

        elif args.testcase == 'bonus':
            self.testcase_list = ['bonus']
            self.img_pth_list = [bonus_pth]

    def detect_featrues(self, img):
        sift = cv2.SIFT_create()
        kpts, features = sift.detectAndCompute(img, None)

        return kpts, features

    def KNN(self, kpts1, kpts2, feature1, feature2):
        matches, good_matches, good_matches_pos = [], [], []

        # find min-distance kpts pairs and second min-distance kpts pairs
        for i in trange(len(feature1)):
            min1_kpts_tuple, min2_kpts_tuple = (None, np.inf), (None, np.inf)

            for j in range(len(feature2)):
                dist = np.linalg.norm(feature1[i] - feature2[j])

                if (dist < min1_kpts_tuple[1]):
                    min2_kpts_tuple = min1_kpts_tuple
                    min1_kpts_tuple = (kpts2[j], dist)
                elif ((dist < min2_kpts_tuple[1]) and (dist > min1_kpts_tuple[1])):
                    min2_kpts_tuple = (kpts2[j], dist)

            matches.append([min1_kpts_tuple, min2_kpts_tuple])

        # ratio test
        for k in range(len(matches)):
            d1 = matches[k][0][1]
            d2 = matches[k][1][1]

            if d1 < 0.75 * d2:
                good_matches.append([kpts1[k], matches[k][0][0]])

        # store kpts coordinates
        for (match1, match2) in good_matches:
            pos1 = (int(match1.pt[0]), int(match1.pt[1]))
            pos2 = (int(match2.pt[0]), int(match2.pt[1]))
            good_matches_pos.append([pos1, pos2])
        
        good_matches_pos = np.transpose(np.array(good_matches_pos), (1, 0, 2))

        return good_matches_pos

    def solve_homography(self, P, m):
        A = []
        for r in range(len(P)): 
            A.append([-P[r, 0], -P[r, 1], -1, 0, 0, 0, P[r, 0] * m[r , 0], P[r , 1] * m[r, 0], m[r, 0]])
            A.append([0, 0, 0, -P[r, 0], -P[r, 1], -1, P[r, 0] * m[r , 1], P[r , 1] * m[r, 1], m[r, 1]])

        # Solve s ystem of linear equations Ah = 0 using SVD
        u, s, vt = np.linalg.svd(A)
        
        # pick H from last line of vt  
        H = np.reshape(vt[8], (3, 3))
        
        # normalization, let H[2,2] equals to 1
        H = (1 / H.item(8)) * H

        return H

    def RANSAC(self, matches_pos):
        THRESHOLD = 5.0
        NUM_ITERATIONS = 5000
        NUM_RANDOM_SAMPLES = 4

        num_samples = matches_pos.shape[1]
        max_inliners = 0
        best_H = None
        
        for _ in trange(NUM_ITERATIONS):
            # sample kpts for computing RANSAC
            sample_idx = random.sample(range(num_samples), NUM_RANDOM_SAMPLES)
            mask_validate = np.ones(num_samples, dtype=bool)
            mask_validate[sample_idx] = False

            # mask out sample kpts for validate inliners
            pos1_validate = matches_pos[0, mask_validate, :].copy()
            pos2_validate = matches_pos[1, mask_validate, :].copy()
            pos1_validate = np.hstack([pos1_validate, np.ones((num_samples - NUM_RANDOM_SAMPLES, 1))]).T
            pos2_validate = np.hstack([pos2_validate, np.ones((num_samples - NUM_RANDOM_SAMPLES, 1))]).T

            # solve homography based on sampled kpts
            cur_H = self.solve_homography(matches_pos[0, sample_idx], matches_pos[1, sample_idx])

            # compute inliners
            pos2_estimate = cur_H @ pos1_validate
            nonzero_idx = (pos2_estimate[2] > 0) # avoid divide zero number

            pos2_estimate = pos2_estimate[:, nonzero_idx]
            pos2_estimate = (pos2_estimate / np.tile(pos2_estimate[2], (3, 1))).T

            pos2_validate = pos2_validate[:, nonzero_idx].T

            dist = np.linalg.norm((pos2_validate[:, :2] - pos2_estimate[:, :2]), axis=1)
            num_inliners = np.sum(dist < THRESHOLD)

            # update H if more inliners founded
            if num_inliners > max_inliners:
                max_inliners = num_inliners
                best_H = cur_H

        return best_H

    def warp(self, img1, img2, H):
        h1,w1 = img1.shape[:2]
        h2,w2 = img2.shape[:2]

        pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
        pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
        pts1_ = cv2.perspectiveTransform(pts1, H)
        pts = np.concatenate((pts1_, pts2), axis=0)

        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

        t = [-xmin,-ymin]
        Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
        A = Ht @ H

        size = (xmax-xmin, ymax-ymin)
        result = cv2.warpPerspective(src=img1, M=A, dsize=size)
        result[t[1]:h2+t[1],t[0]:w2+t[0]] = img2

        return result

    def run(self):
        for (testcase, image_pair) in zip(self.testcase_list, self.img_pth_list):
            print('Testcase: {}'.format(testcase))

            img1, img1_gray = read_img(image_pair[0])

            for i in range(1, len(image_pair)):
                img2, img2_gray = read_img(image_pair[i])

                print()
                print('-' * 20 + ' Stiching with m{} '.format(i + 1) + '-' * 20)
                print('Step1: Detecting keypoints (feature) on the images')
                kpts1, feature1 = self.detect_featrues(img1_gray)
                kpts2, feature2 = self.detect_featrues(img2_gray)

                print('Step2: Finding features correspondences (feature matching)')
                good_matches = self.KNN(kpts1, kpts2, feature1, feature2)

                print('Step3: Computing homography matrix')
                bestH = self.RANSAC(good_matches)

                print('Step4: Warp images')
                result = self.warp(img1, img2, bestH)
                
                img1, img1_gray = result, img_to_gray(result)

            cv2.imwrite("{}.jpg".format(testcase), img1)
            # creat_im_window("result_m"+str(cnt)+"_m"+str(cnt+3),img1)
            # im_show()

if __name__ == '__main__':
    args = parse_args()
    assert args.testcase in ['baseline', 'bonus']

    sticher = ImageStiching(args)
    sticher.run()
