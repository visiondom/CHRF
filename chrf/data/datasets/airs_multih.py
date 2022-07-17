import os
from os.path import join as opj

# 100 models variants, 70 families, 30 makers
hier_map = {0: "model", 1: "family", 2: "maker"}

trees = [
[1, 1, 1],
[2, 2, 1],
[3, 3, 1],
[4, 3, 1],
[5, 3, 1],
[6, 3, 1],
[7, 4, 1],
[8, 4, 1],
[9, 5, 1],
[10, 5, 1],
[11, 5, 1],
[12, 5, 1],
[13, 6, 1],
[14, 7, 2],
[15, 8, 3],
[16, 9, 3],
[17, 10, 7],
[18, 10, 7],
[19, 11, 7],
[20, 12, 4],
[21, 13, 5],
[22, 14, 5],
[23, 15, 5],
[24, 16, 5],
[25, 16, 5],
[26, 16, 5],
[27, 16, 5],
[28, 16, 5],
[29, 16, 5],
[30, 16, 5],
[31, 16, 5],
[32, 17, 5],
[33, 17, 5],
[34, 17, 5],
[35, 17, 5],
[36, 18, 5],
[37, 18, 5],
[38, 19, 5],
[39, 19, 5],
[40, 19, 5],
[41, 20, 5],
[42, 20, 5],
[43, 21, 21],
[44, 22, 14],
[45, 23, 9],
[46, 24, 9],
[47, 25, 9],
[48, 25, 9],
[49, 26, 8],
[50, 27, 8],
[51, 28, 8],
[52, 28, 8],
[53, 29, 12],
[54, 29, 12],
[55, 30, 23],
[56, 31, 14],
[57, 32, 14],
[58, 33, 14],
[59, 34, 23],
[60, 35, 12],
[61, 36, 12],
[62, 37, 12],
[63, 38, 13],
[64, 39, 26],
[65, 40, 15],
[66, 41, 15],
[67, 41, 15],
[68, 41, 15],
[69, 42, 15],
[70, 42, 15],
[71, 43, 15],
[72, 44, 16],
[73, 45, 23],
[74, 46, 22],
[75, 47, 11],
[76, 48, 11],
[77, 49, 18],
[78, 50, 18],
[79, 51, 18],
[80, 52, 6],
[81, 53, 19],
[82, 53, 19],
[83, 54, 7],
[84, 55, 20],
[85, 56, 4],
[86, 57, 21],
[87, 58, 23],
[88, 59, 23],
[89, 59, 23],
[90, 60, 23],
[91, 61, 17],
[92, 62, 25],
[93, 63, 27],
[94, 64, 27],
[95, 65, 28],
[96, 66, 10],
[97, 67, 24],
[98, 68, 29],
[99, 69, 29],
[100, 70, 30]
]

# 6667 images are used for training and 3333 images for testing
def load_airs_data_multih(dataset_dir):
    img_dir = opj(dataset_dir, "images")

    FILENAME_LENGTH = 7

    variants_dict = {}
    with open(opj(dataset_dir, 'variants.txt'), 'r') as f:
        for idx, line in enumerate(f.readlines()):
            variants_dict[line.strip()] = idx
    num_classes = len(variants_dict)

    train_list_path = opj(dataset_dir, 'images_variant_trainval.txt')
    train_set_dict = []
    with open(train_list_path, 'r') as f:
        for line in f.readlines():
            fname_and_variant = line.strip()
            img_path = opj(img_dir, '%s.jpg' % fname_and_variant[:FILENAME_LENGTH])
            img_dict = {
                "img": img_path,
                "category": fname_and_variant[FILENAME_LENGTH + 1:],
            }
            label = variants_dict[fname_and_variant[FILENAME_LENGTH + 1:]]
            h_labels = {
                h_name: trees[label][hier] - 1 for hier, h_name in hier_map.items()
            }
            assert label == h_labels["model"]
            img_dict.update(h_labels)
            train_set_dict.append(img_dict)

    test_list_path = opj(dataset_dir, 'images_variant_test.txt')
    test_set_dict = []
    with open(test_list_path, 'r') as f:
        for line in f.readlines():
            fname_and_variant = line.strip()
            img_path = opj(img_dir, '%s.jpg' % fname_and_variant[:FILENAME_LENGTH])
            img_dict = {
                "img": img_path,
                "category": fname_and_variant[FILENAME_LENGTH + 1:],
            }
            label = variants_dict[fname_and_variant[FILENAME_LENGTH + 1:]]
            h_labels = {
                h_name: trees[label][hier] - 1 for hier, h_name in hier_map.items()
            }
            assert label == h_labels["model"]
            img_dict.update(h_labels)
            test_set_dict.append(img_dict)

        return train_set_dict, test_set_dict