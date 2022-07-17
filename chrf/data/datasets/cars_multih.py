import os
from os.path import join as opj
from scipy.io import loadmat


# 196 car models, 9 car makers
hier_map = {0: "model", 1: "maker"}

trees = [
[1, 7],
[2, 6],
[3, 6],
[4, 6],
[5, 6],
[6, 3],
[7, 4],
[8, 2],
[9, 3],
[10, 2],
[11, 3],
[12, 2],
[13, 3],
[14, 3],
[15, 3],
[16, 6],
[17, 6],
[18, 9],
[19, 4],
[20, 6],
[21, 2],
[22, 3],
[23, 6],
[24, 6],
[25, 3],
[26, 6],
[27, 2],
[28, 3],
[29, 6],
[30, 9],
[31, 2],
[32, 7],
[33, 7],
[34, 3],
[35, 6],
[36, 2],
[37, 7],
[38, 2],
[39, 2],
[40, 6],
[41, 6],
[42, 3],
[43, 3],
[44, 6],
[45, 2],
[46, 3],
[47, 6],
[48, 7],
[49, 6],
[50, 7],
[51, 6],
[52, 7],
[53, 1],
[54, 1],
[55, 2],
[56, 3],
[57, 3],
[58, 7],
[59, 2],
[60, 5],
[61, 6],
[62, 7],
[63, 6],
[64, 8],
[65, 1],
[66, 3],
[67, 6],
[68, 7],
[69, 1],
[70, 1],
[71, 8],
[72, 3],
[73, 6],
[74, 1],
[75, 1],
[76, 7],
[77, 2],
[78, 5],
[79, 6],
[80, 2],
[81, 2],
[82, 9],
[83, 9],
[84, 9],
[85, 5],
[86, 1],
[87, 1],
[88, 8],
[89, 7],
[90, 1],
[91, 1],
[92, 9],
[93, 3],
[94, 7],
[95, 7],
[96, 6],
[97, 6],
[98, 4],
[99, 3],
[100, 2],
[101, 3],
[102, 2],
[103, 2],
[104, 3],
[105, 6],
[106, 1],
[107, 2],
[108, 5],
[109, 7],
[110, 7],
[111, 1],
[112, 3],
[113, 1],
[114, 1],
[115, 6],
[116, 9],
[117, 6],
[118, 7],
[119, 8],
[120, 7],
[121, 7],
[122, 1],
[123, 2],
[124, 1],
[125, 1],
[126, 5],
[127, 5],
[128, 3],
[129, 6],
[130, 4],
[131, 7],
[132, 7],
[133, 7],
[134, 6],
[135, 6],
[136, 6],
[137, 6],
[138, 6],
[139, 4],
[140, 6],
[141, 3],
[142, 7],
[143, 7],
[144, 3],
[145, 7],
[146, 7],
[147, 7],
[148, 7],
[149, 7],
[150, 3],
[151, 3],
[152, 3],
[153, 3],
[154, 7],
[155, 7],
[156, 6],
[157, 2],
[158, 2],
[159, 7],
[160, 3],
[161, 2],
[162, 6],
[163, 3],
[164, 6],
[165, 6],
[166, 8],
[167, 6],
[168, 4],
[169, 8],
[170, 4],
[171, 3],
[172, 3],
[173, 6],
[174, 5],
[175, 2],
[176, 6],
[177, 6],
[178, 4],
[179, 2],
[180, 3],
[181, 6],
[182, 6],
[183, 4],
[184, 6],
[185, 6],
[186, 7],
[187, 6],
[188, 6],
[189, 7],
[190, 4],
[191, 4],
[192, 4],
[193, 4],
[194, 6],
[195, 7],
[196, 2]
]

# 8,144 images are used for training and 8,041 images for testing
def load_cars_data_multih(dataset_dir):
    img_dir = dataset_dir

    list_path = opj(dataset_dir, 'cars_annos.mat')
    list_mat = loadmat(list_path)
    num_inst = len(list_mat['annotations']['relative_im_path'][0])

    train_set_dict = []
    test_set_dict = []

    for i in range(num_inst):
        img_path = list_mat['annotations']['relative_im_path'][0][i].item()
        img_dict = {
            "img": opj(img_dir, img_path),
        }
        label = list_mat['annotations']['class'][0][i].item()
        h_labels = {
           h_name: trees[label-1][hier]-1 for hier, h_name in hier_map.items()
        }
        assert (label-1) == h_labels["model"]
        img_dict.update(h_labels)

        if list_mat['annotations']['test'][0][i].item() == 0:
            train_set_dict.append(img_dict)
        elif list_mat['annotations']['test'][0][i].item() == 1:
            test_set_dict.append(img_dict)
    return train_set_dict, test_set_dict