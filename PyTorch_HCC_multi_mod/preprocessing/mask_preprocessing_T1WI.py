import os
import shutil
import nibabel as nib
import nibabel.processing as nibproc
import numpy as np
import concurrent.futures
import SimpleITK as sitk
import matplotlib.pyplot as plt


def slice_select(roi_img):
    # print('sum', np.sum(roi_img))
    sum_all = []
    idx_all = []
    for idx in range(roi_img.shape[2]):
        if np.sum(roi_img[:, :, idx]) > 0:
            # print('idx', idx)
            sum_all.append(np.sum(roi_img[:, :, idx]))
            idx_all.append(idx)
    print('sum_all', len(sum_all))
    # print('sum_all.index(', sum_all.index(max(sum_all)))
    if len(sum_all) > 0:
        max_idx = sum_all.index(max(sum_all))
        slice_num = idx_all[max_idx]
        print('slice_num', slice_num)
    else:
        slice_num = 0
    # print('slice_num, idx_all', slice_num, idx_all)
    return slice_num, idx_all


def resampling_reorientation(name, mod, img, output, out_shape):
    # resampling to 240x240x155 1mm3
    resampledIm = nibproc.conform(img, out_shape=out_shape, voxel_size=(1.0, 1.0, 1.0), order=1, cval=0.0)  #双线性插值重新采样
    # Modify metadata for reorientation
    n1_header = img.header
    # print(n1_header)
    n1_header.set_sform(np.diag([0, 0, 0, 0]), code='unknown')
    qform = np.array(
        [[1., 0., 0., -120.],
         [0., 1., 0., -129.],
         [0., 0., 1, -68.],
         [0., 0., 0., 1.]])

    n1_header.set_qform(qform, code='scanner')
    # print('--------------------------')
    # print(n1_header)
    img_data = img.get_data()
    print('img_data', img_data.shape, np.sum(img_data))

    # Reoriented to SRI24 t1 Atlas
    new_img = nib.nifti1.Nifti1Image(resampledIm.get_fdata(), None, header=n1_header)
    # print('new_img', new_img.get_data())
    show_img = new_img.get_data()
    # print('type', type(show_img))
    print('show_img', show_img.shape, np.sum(show_img))

    slice_num, idx_all = slice_select(show_img)
    if len(idx_all)>0 and slice_num > 0:
        plt.imshow(show_img[:, :, slice_num], cmap='gray')
        plt.title('resample' + str(slice_num))
        plt.savefig(os.path.join('/data/Wendy/HCC/resample_mask_png', name + '_' + mod + '_resample.png'))
        plt.close()
        slice_num, idx_all = slice_select(img_data)
        plt.imshow(img_data[:, :, slice_num], cmap='gray')
        plt.title('ori' + str(slice_num))
        plt.savefig(os.path.join('/data/Wendy/HCC/resample_mask_png', name + '_' + mod + '_ori.png'))
        plt.close()
        nib.save(new_img, output)
        return True
    else:
        print('not satisfied')
        print(name)
        return False


def move_file(src_dir, tar_dir):
    if not os.path.exists(tar_dir):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(tar_dir)
    shutil.copy(src_dir, tar_dir)


def processing(src_file):
    tar_file = os.path.join(path_output, mod + '_' + src_file.split('/')[-1])
    print('tar_file', tar_file)
    # if not os.path.exists(tar_file):
    if len(src_file) > 0:
        img = nib.load(src_file)
        array_data = img.get_data()
        # print('sum', np.sum(array_data))
        # print('array_data', array_data.shape)
        # uni = list(np.unique(array_data))
        # print('uni', uni)
        array_img = nib.Nifti1Image(array_data, img.affine)
        #
        # # img_npy = np.array(img)
        # # print('img', np.sum(img_npy))
        # if os.path.exists(tar_file):
        #     pass
        # else:
        # print(tar_file)

        satistied_status = resampling_reorientation(src_file.split('/')[-1].replace('.nii.gz', ''), mod, array_img, tar_file, out_shape)
        if not satistied_status:
            mask_not_satisfied.append(src_file.split('/')[-1])
        print('mask_not_satisfied', mask_not_satisfied)



path = '/data/Wendy/HCC/HCC'
path_output = '/data/Wendy/HCC/resample_mask'

mod = 'T1WI'

path_AP = os.path.join(path, mod)
patients = os.listdir(path_AP)
out_shape = (512, 512, 150)
# img startwith data, mask startwith mask
patients_AP = []
for i in patients:
    file_p = os.path.join(path_AP, i)
    src_file = [os.path.join(file_p, o) for o in os.listdir(file_p) if o.startswith('mask')][0]
    print('src_file', src_file)
    patients_AP.append(src_file)

mask_not_satisfied = []
# executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
# executor.map(processing, patients_AP)
#
print('patients_AP', len(patients_AP), patients_AP)

for i in patients_AP:
    # if '10149752' in i:
        print('i', i)
        processing(i)

# mask_not_satisfied ['mask_10258812.nii.gz', 'mask_10149752.nii.gz', 'mask_10208231.nii.gz']