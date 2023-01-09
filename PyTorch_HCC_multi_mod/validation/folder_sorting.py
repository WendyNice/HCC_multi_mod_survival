import os
import shutil
import SimpleITK as sitk


def move_file(src_dir, tar_dir):
    if not os.path.exists(tar_dir):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(tar_dir)
    shutil.copy(src_dir, tar_dir)


def dicom_to_nii(patient_name_path):
    print('patient_name_path', patient_name_path)
    reader = sitk.ImageSeriesReader()
    slice_names = reader.GetGDCMSeriesFileNames(patient_name_path)
    print('slice_names', slice_names)
    #slice_names=sorted(slice_names)
    reader.SetFileNames(slice_names)
    image = reader.Execute()

    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    # keys = image.GetMetaDataKeys()

    image_array = sitk.GetArrayFromImage(image)  # (Depth, Height, Width)
    out = sitk.GetImageFromArray(image_array)
    out.SetOrigin(origin)
    out.SetSpacing(spacing)
    return out, spacing, image_array.shape


def save_img_roi(image_paths, roi_paths, modality, mod_name='AP'):
    path_mod = os.path.join(image_paths, j, modality)
    if not os.path.exists(path_mod):
        path_mod = path_mod.replace(modality, mod_name.replace('PVP', 'VP'))
        # print('path_mod', path_mod)
    img_tar_file = os.path.join(path_output, j, mod_name, 'data.nii.gz')
    if not os.path.exists(os.path.join(path_output, j, mod_name)):
        os.makedirs(os.path.join(path_output, j, mod_name))
    img_nii, _, _ = dicom_to_nii(path_mod)
    sitk.WriteImage(img_nii, img_tar_file)
    roi_tar_file = os.path.join(path_output, j, mod_name)
    roi_src_file = os.path.join(roi_paths, j, str(j) + mod_name.replace('WI', '') + '.nii.gz')
    move_file(roi_src_file, roi_tar_file)


path = '/data/Wendy/HCC/valid_set'
path_output = '/data/Wendy/HCC/valid_set/data_sort'

AP = '4_2'
PVP = '4_3'
T1WI = '4_1'
# T2WI = 'T2WI'
image_paths = os.path.join(path, 'image')
roi_paths = os.path.join(path, 'roi')
patient_ls = os.listdir(image_paths)
for j in patient_ls:
    save_img_roi(image_paths, roi_paths, AP, mod_name='AP')
    save_img_roi(image_paths, roi_paths, PVP, mod_name='PVP')
    save_img_roi(image_paths, roi_paths, T1WI, mod_name='T1WI')



