import cv2
import os
import argparse
import numpy as np


def get_roi(mask_list):
    box_list = []
    for mask in mask_list:
        _, contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        box = cv2.boundingRect(contours[0])
        box_list.append(box)
    # x, y, w, h = cv2.boundingRect(contours[0])
    # mask_roi = mask[y: y + h, x: x + w]
    # src_roi = src[y: y + h, x: x + w]

    return box_list


def get_chartlet(bg, chartlet_label, src, mask, box, cur_class_laebl):
    rx, ry, rw, rh = box
    bg_height, bg_width = bg.shape[:2]
    limit_height = bg_height - rh
    limit_width = bg_width - rw

    start_x = np.random.randint(0, limit_width)
    start_y = np.random.randint(0, limit_height)

    mask_roi = mask[ry: ry + rh, rx: rx + rw]
    src_roi = src[ry:ry + rh, rx: rx + rw]
    bg_roi = bg[start_y: start_y + rh, start_x: start_x + rw]
    cl_roi = chartlet_label[start_y: start_y + rh, start_x: start_x + rw]

    cl_tmp = np.zeros_like(cl_roi)
    cl_tmp += int(cur_class_laebl)
    cl_roi = cv2.bitwise_and(np.uint16(mask_roi), cl_tmp)
    chartlet_label[start_y: start_y + rh, start_x: start_x + rw] = cl_roi

    mask_roi = cv2.merge([mask_roi, mask_roi, mask_roi])
    mask_roi_not = cv2.bitwise_not(mask_roi)

    # bg_roi = cv2.bitwise_and(bg_roi, mask_roi_not) + \
    #     cv2.bitwise_and(src_roi, mask_roi)
    mask_roi_f = np.float32(mask_roi / 255)
    mask_roi_not_f = np.float32(mask_roi_not / 255)
    mask_roi = cv2.GaussianBlur(mask_roi_f, (5, 5), 15)
    mask_roi_not = cv2.GaussianBlur(mask_roi_not_f, (5, 5), 15)
    bg_roi = np.multiply(bg_roi, mask_roi_not_f) + \
        np.multiply(src_roi, mask_roi_f)
    # cv2.waitKey()

    bg[start_y: start_y + rh, start_x: start_x + rw] = bg_roi
    # cv2.imshow('bg', bg)
    # cv2.imshow('br', bg_roi)
    # cv2.waitKey()

    return bg, chartlet_label


def get_obj_list(src_path, mask_path, name_list, cur_obj_idx):
    src_list = []
    mask_list = []
    obj_class_name = []

    for idx in cur_obj_idx:
        cur_src = cv2.imread(os.path.join(
            src_path, name_list[idx]), cv2.IMREAD_UNCHANGED)
        cur_mask = cv2.imread(os.path.join(
            mask_path, name_list[idx]), cv2.IMREAD_UNCHANGED)
        cur_mask_u8 = np.uint8(cur_mask) * 255
        cur_obj_class_name = name_list[idx].split('.')[0].split('-')[0]

        src_list.append(cur_src)
        mask_list.append(cur_mask_u8)
        obj_class_name.append(cur_obj_class_name)

    return src_list, mask_list, obj_class_name


def get_label_map(label_map_path):
    label_map = dict()
    cnt = 1
    with open(label_map_path, 'r') as fd:
        while True:
            class_name = fd.readline().strip()
            if class_name is None or class_name is "":
                break
            if not class_name in label_map:
                label_map[class_name] = cnt
                cnt += 1

    return label_map


def main():
    # mask_path = input('enter mask image path\n>')
    # src_path = input('enter source image path\n>')

    # mask_path = r'D:\ImgPro\Project\competition\dataset\label\label'
    # src_path = r'D:\ImgPro\Project\competition\dataset\label\src'
    # bg_path = r'D:\ImgPro\Project\competition\dataset\label\bg'

    parser = argparse.ArgumentParser(
        description='command option for this script')
    parser.add_argument('--src', type=str, default=None)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--bg', type=str, default=None)
    parser.add_argument('--label_map', type=str, default=None)
    parser.add_argument('--output', type=str, default=r'.\chartlet')
    parser.add_argument('--num', type=int, default=1000)

    args = parser.parse_args()
    src_path = args.src
    mask_path = args.mask
    bg_path = args.bg
    label_map_path = args.label_map
    output_path = args.output
    gen_num = args.num

    output_img_path = os.path.join(output_path, 'src')
    output_label_path = os.path.join(output_path, 'label')
    if not os.path.exists(output_img_path):
        os.makedirs(output_img_path)
    if not os.path.exists(output_label_path):
        os.makedirs(output_label_path)

    if src_path is None or mask_path is None \
            or bg_path is None or label_map_path is None:
        print(parser.print_help())
        exit()

    label_map = get_label_map(label_map_path)
    print(label_map)
    name_list = list(set(os.listdir(src_path)))
    bg_name_list = list(set(os.listdir(bg_path)))
    obj_total_num = len(name_list)

    per_bg_pic = gen_num // len(bg_name_list)
    print('per_bg_pic:', per_bg_pic)

    for i in range(gen_num):
        cur_bg_num = i // per_bg_pic
        bg = cv2.imread(os.path.join(
            bg_path, bg_name_list[cur_bg_num % len(bg_name_list)]), cv2.IMREAD_UNCHANGED)

        cur_obj_num = np.random.randint(1, 5)
        cur_obj_idx = np.random.randint(0, obj_total_num, cur_obj_num)
        src_list, mask_list, obj_class_name = get_obj_list(
            src_path, mask_path, name_list, cur_obj_idx)

        box_list = get_roi(mask_list)
        chartlet = bg
        chartlet_label = np.zeros_like(bg[:, :, 0])
        chartlet_label = np.uint16(chartlet_label)
        for k in range(len(box_list)):
            src = src_list[k]
            box = box_list[k]
            mask = mask_list[k]
            cur_class_name = obj_class_name[k]
            chartlet, chartlet_label = get_chartlet(
                chartlet, chartlet_label, src, mask, box, label_map[cur_class_name])

        # cv2.imshow('c', chartlet)
        # cv2.waitKey()
        cv2.imwrite(os.path.join(output_img_path,
                                 str(i) + '.png'), chartlet)
        cv2.imwrite(os.path.join(output_label_path,
                                 str(i) + '.png'), chartlet_label)
        print('chartlet #%d pic in path: %s' %
              (i, os.path.join(output_path, str(i) + '.png')))


if __name__ == '__main__':
    main()
