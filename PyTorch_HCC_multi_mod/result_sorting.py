import os

path = r'/home/amax/Wendy/SYSU-hcc/data_clf_2/3d_result'
files = os.listdir(path)
files.sort()
print('files', files)

for i in files:
    file_path = os.path.join(path, i)
    # print('file_path', file_path)
    file_matrix = os.path.join(file_path, 'confusion_matrix')


    matrix_path = os.listdir(file_matrix)

    auc_list = [float(i.split('.png')[0]) for i in matrix_path if i.startswith('0')]
    auc_list.sort()

    if len(auc_list) > 0:
        print(file_matrix.split('/')[-2].replace('output_', ''))
        # print('auc_list', auc_list)
        auc_max = auc_list[-1]
        print('AUC', auc_max)
    else:
        print('############################################')
        print(print('file_matrix', file_matrix))


