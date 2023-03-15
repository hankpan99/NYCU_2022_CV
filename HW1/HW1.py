import os
import cv2
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

image_row = 120
image_col = 120

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image_row, image_col = image.shape
    return image

# my functions
def parse_args():
    parser = argparse.ArgumentParser(description='argument settings')
    parser.add_argument('-nv', '--novisualize', help='type = "bool", disable visualization', action='store_true')
    parser.add_argument('-t', '--testcase', help='type = "str", specify the test case', type=str, default=None)

    args = parser.parse_args()

    return args

def load_data(test_case):
    with open(f'./test/{test_case}/LightSource.txt', 'r') as f:
        file_rows = list(filter(None, f.read().replace('(', '').replace(')', '').split('\n')))

    image_list, light_source_list = [], []
    for row in file_rows:
        items = row.split(': ')
        
        image_list.append(read_bmp(f'test/{test_case}/{items[0]}.bmp'))
        light_source_list.append(np.array([float(value) for value in items[1].split(',')]))

    return image_list, light_source_list

def normal_estimation(image_list, light_source_list):
    intensity_arr = np.zeros((len(image_list), image_row * image_col))
    light_arr = np.zeros((len(image_list), 3))

    for cnt, (image, light) in enumerate(zip(image_list, light_source_list)):
        intensity_arr[cnt, :] = image.flatten()
        light_arr[cnt, :] = light / np.linalg.norm(light)

    raw_normal_arr = np.linalg.inv(light_arr.T @ light_arr) @ light_arr.T @ intensity_arr
    vector_length = np.linalg.norm(raw_normal_arr, axis=0)
    vector_length[vector_length == 0] = 1

    normal_map = raw_normal_arr / np.tile(vector_length, (3, 1))

    return normal_map.T

def remove_outlier(depth_map):
    mean, std = np.mean(depth_map), np.std(depth_map)
    zscore = (depth_map - mean) / std
    depth_map[np.abs(zscore) > 2.4] = 0

    return depth_map

def surface_reconstruction(test_case, images, normal_map):
    normal_map = np.reshape(normal_map, (image_row, image_col, 3))
    roi_mask = np.copy(np.reshape(images, (image_row, image_col)))
    roi_h, roi_w = np.where(roi_mask != 0)

    NUM_PIXEL = roi_h.shape[0]

    pixel_2_maskid = np.zeros((image_row, image_col)) 
    for i in range(NUM_PIXEL):
        pixel_2_maskid[roi_h[i], roi_w[i]] = i

    M = lil_matrix((2 * NUM_PIXEL, NUM_PIXEL))
    V = np.zeros((2 * NUM_PIXEL, 1))

    for i in range(NUM_PIXEL):
        h = roi_h[i]
        w = roi_w[i]

        Nx = normal_map[h, w, 0]
        Ny = normal_map[h, w, 1]
        Nz = normal_map[h, w, 2]

        i_row = i
        if roi_mask[h, w + 1]:
            i_col = pixel_2_maskid[h, w + 1]
            M[i_row, i] = -1
            M[i_row, i_col] = 1
            V[i_row] = -Nx / Nz
        elif roi_mask[h, w - 1]:
            i_col = pixel_2_maskid[h, w - 1]
            M[i_row, i_col] = -1
            M[i_row, i] = 1
            V[i_row] = -Nx / Nz
        
        i_row = i + NUM_PIXEL
        if roi_mask[h + 1, w]:
            i_col = pixel_2_maskid[h + 1, w]
            M[i_row, i] = -1
            M[i_row, i_col] = 1
            V[i_row] = Ny / Nz
        elif roi_mask[h - 1, w]:
            i_col = pixel_2_maskid[h - 1, w]
            M[i_row, i_col] = -1
            M[i_row, i] = 1
            V[i_row] = Ny / Nz

    depth_map_wmask = spsolve(M.T @ M, M.T @ V)
    depth_map = np.zeros((image_row, image_col))

    for i in range(NUM_PIXEL):
        h = roi_h[i]
        w = roi_w[i]
        depth_map[h, w] = depth_map_wmask[i]

    if test_case == 'venus':
        depth_map = remove_outlier(depth_map)
        
    return depth_map

def save_answer(test_case, normal_map, depth_map):
    dst_pth = f'./answer/{test_case}/'

    if not os.path.exists(dst_pth):
        os.makedirs(dst_pth)

    with open(dst_pth + 'normal_map.npy', 'wb') as f:
        np.save(f, normal_map)

    with open(dst_pth + 'depth_map.npy', 'wb') as f:
        np.save(f, depth_map)

    save_ply(depth_map, dst_pth + 'pc.ply')

def visualize_all(test_case, normal_map, depth_map):
    normal_visualization(normal_map)
    depth_visualization(depth_map)
    show_ply(f'./answer/{test_case}/pc.ply')

if __name__ == '__main__':

    test_case_list = ['bunny', 'star', 'venus']

    args = parse_args()
    if args.testcase != None:
        assert args.testcase in test_case_list
        test_case_list = [args.testcase]

    for test_case in test_case_list:
        # load data
        image_list, light_source_list = load_data(test_case)

        # normal estimation
        normal_map = normal_estimation(image_list, light_source_list)

        # surface reconstruction
        depth_map = surface_reconstruction(test_case, image_list[0], normal_map)

        # save answer
        save_answer(test_case, normal_map, depth_map)

        # visualize
        if not args.novisualize:
            visualize_all(test_case, normal_map, depth_map)