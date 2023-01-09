

from collections import OrderedDict
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
import numpy as np
from nnunet.configuration import default_num_threads
from scipy.ndimage import label
from sklearn.model_selection import train_test_split
import pandas as pd
import nibabel as nib
import nibabel.processing as nibproc
from dipy.io.image import load_nifti, save_nifti
# from intensity_normalization.normalize import zscore
from dipy.align.imaffine import transform_centers_of_mass, AffineMap, MutualInformationMetric, AffineRegistration
from dipy.align.transforms import TranslationTransform3D, RigidTransform3D, AffineTransform3D
import dicom2nifti as dcmnii
import time
import os
import shutil
import matplotlib.pyplot as plt
import time
import concurrent.futures


def dicom_to_nii(patient_name_path):
    reader = sitk.ImageSeriesReader()
    slice_names = reader.GetGDCMSeriesFileNames(patient_name_path)

    print(slice_names)

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
    return out


def resampling_reorientation(img, output, out_shape):
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

    # Reoriented to SRI24 t1 Atlas
    new_img = nib.nifti1.Nifti1Image(resampledIm.get_fdata(), None, header=n1_header)
    # print('new_img', new_img.get_data())
    # show_img = new_img.get_data()
    # # print('type', type(show_img))
    # # print('show_img', show_img.shape)
    # #
    # plt.imshow(show_img[:, :, 50], cmap='gray')
    # plt.show()
    nib.save(new_img, output)


# Bias Correction
def N4BiasCorrection(imDir, output):
    # Read image define corrector
    image = sitk.ReadImage(imDir, sitk.sitkFloat32)
    print(type(image))
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    # Create mask
    mask = (image > 0)

    # Execute corrector
    corrector.SetMaximumNumberOfIterations(np.array([200], dtype='int').tolist())
    corrected_image = corrector.Execute(image, mask)
    print(type(corrected_image))

    sitk.WriteImage(corrected_image, output)


def rigidRegistration(static, moving, output):
    # Load static image
    static_data, static_affine, static_img = load_nifti(static, return_img=True)

    # load moving image
    moving_data, moving_affine, moving_img = load_nifti(moving, return_img=True)
    # print(1)
    # traslattion mass center
    c_of_mass = transform_centers_of_mass(static_data, static_affine, moving_data, moving_affine)
    # transformed = c_of_mass.transform(moving_data)
    starting_affine = c_of_mass.affine
    # print(2)

    # elements for registration
    nbins = 32
    sampling_prop = 5
    metric = MutualInformationMetric(nbins, sampling_prop)
    # print('2.5')
    level_iters = [1]
    sigmas = [0.0]
    factors = [1]

    # Rigid registration
    affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)
    # print('2.7')
    transform = RigidTransform3D()
    params0 = None
    rigid = affreg.optimize(static_data, moving_data, transform, params0,
                            static_affine, moving_affine,
                            starting_affine=starting_affine)
    # print(3)
    transformed = rigid.transform(moving_data)
    transformed = transformed.astype(np.uint16)
    save_nifti(output, transformed, static_affine)
    return rigid


def processing(case_path):
    start_time = time.time()
    # print('processed', processed)
    # if case_path not in processed:
        # print(case_path + ' have been processed')
    # else:
    try:
        print('#' * 100)
        caseTest = case_path
        print('caseTest', caseTest)

        pcase = os.path.join(pdata, caseTest)
        print('pcase', pcase)

        pcaseo = os.path.join(output_folder, caseTest)
        print('pcaseo', pcaseo)
        if not os.path.exists(pcaseo):
            os.makedirs(pcaseo)

        for modality in modalities:
            pcaseMod = pcase + '/' + modality
            pcaseoMod = pcaseo + '/' + modality
            print('pcaseMod', pcaseMod)
            print('pcaseoMod', pcaseoMod)
            if not os.path.exists(pcaseoMod):
                os.makedirs(pcaseoMod)
            # dcmnii.convert_directory(pcaseMod, pcaseoMod, compression=True)
            src_file = [os.path.join(pcaseMod, o) for o in os.listdir(pcaseMod) if o.startswith('data')][0]
            shutil.copy(src_file, pcaseoMod)
            # print('built ' + pcaseoMod + '/' + modality)

        for i in range(len(modalities)):
            # Load image
            pdata_mod = os.path.join(pcaseo, modalities[i])
            print('pdata_mod', pdata_mod)
            listFiles = os.listdir(pdata_mod)[0]
            print('listFiles', listFiles)
            pdata_new = os.path.join(pdata_mod, listFiles)
            print('pdata_new', pdata_new)
            # img = nib.load(pdata_new)
            # img = nib.Nifti1Image.from_filename(pdata_new)
            # print('img', img.shape)


            # os.chdir(pcaseo + modalities[i])
            #
            # resampling to 240x240x155 1mm3
            resam_reor_path = os.path.join(pdata_mod, modalities[i] + '_resam&reor.nii.gz')
            # print('resampling')
            # resampling_reorientation(img, resam_reor_path, out_shape=(512, 512, 150))
            # print('Bias correction')
            # Bias correction
            bias_path = os.path.join(pdata_mod, modalities[i] + '_bias.nii.gz')
            if not os.path.exists(bias_path):
                N4BiasCorrection(resam_reor_path, bias_path)
            # print('Registration')
            # # Registration dipy
            if modalities[i] == 'AP':
                print('Registration for AP')
                output_path = os.path.join(pdata_mod, modalities[i] + '_registered.nii.gz')
                # static = bias_path
                if not os.path.exists(output_path):
                    moving = bias_path
                    output = output_path
                    moving_data, moving_affine, moving_img = load_nifti(moving, return_img=True)
                    save_nifti(output, moving_data, moving_affine)
                    # rigid = rigidRegistration(static, moving, output)
            else:
                print('Registration for other modality')
                output_path = os.path.join(pdata_mod, modalities[i] + '_registered.nii.gz')
                if not os.path.exists(output_path):
                    print('running registration')
                    moving_path = os.path.join(pdata_mod, modalities[i] + '_bias.nii.gz')
                    static = os.path.join(pcaseo, 'AP', 'AP' + '_bias.nii.gz')
                    moving = moving_path
                    output = output_path
                    rigidRegistration(static, moving, output)
    except IndexError as err:
        print("IndexError error: {0}".format(err))
        print('Failed file is %s' % (case_path))
        # failed_files.append(case_path)
        pass
    end_time = time.time()
    print('time', (end_time - start_time) / 60)

if __name__ == "__main__":


    def load_save(args):
        data_file, end_p_id = args
        # print('end_p_id', end_p_id)
        print('data_file', data_file)
        img_itk = dicom_to_nii(data_file)
        img_npy = sitk.GetArrayFromImage(img_itk)
        print('shape', img_npy.shape)
        # print('seg_file', seg_file)
        pat_id = data_file.split("/")[-2]
        pat_id = pat_id
        # print('pat_id', pat_id + end_p_id)
        # sitk.WriteImage(img_itk, join(img_dir_train, pat_id + end_p_id))
        return pat_id


    def select_modality(patients_path, mod):
        img_paths = []
        for path in patients_path:
            # print('train_path', path)
            files = os.listdir(path)
            # print('files', files)
            img_1 = [i for i in files if i == mod][0]
            img_paths.append(os.path.join(path, img_1))
        # print('img_paths', len(img_paths))
        return img_paths

    #
    # for patient in patients_0:
    #     print("##############################################")
    #     print('patient', patient)
    #     mods = ['FLAIR', 'T1w', 'T2w', 'T1wCE']
    #     for mod in mods:
    #         print('mod', mod)
    #         img_mod = os.path.join(patient, mod)
    #         img_itk = dicom_to_nii(img_mod)
    #         img_npy = sitk.GetArrayFromImage(img_itk)
    #         print('shape', img_npy.shape)


    output_folder = '/data/Wendy/HCC/resample'
    # img_dir_train = join(output_folder, "imagesTr")
    processed = os.listdir(output_folder)

    start_time = time.time()
    modalities = ["AP", "PVP", "T1WI", "T2WI"]
    # Conversion to nifti
    pdata = '/data/Wendy/HCC/data_sort'
    casesList = os.listdir(pdata)
    print('casesList', len(casesList))
    # casesList.sort()
    # failed_files = []
    # for case_idx in range(len(casesList)):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
    executor.map(processing, casesList)
    # for case in casesList:
    #     # if case in ['10161476']:
    #         print('case', case)
    #         processing(case)
        # print('############ End Preprocessing ############')

    # print('failed_files', failed_files)

