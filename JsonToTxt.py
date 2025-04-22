import json
import os

# 输入文件目录
# json_path = r"datasets/lables/train"  # 标注数据json的文件地址
# txt_path = r"datasets/lables/train"  # 转换后的txt标签文件存放的文件夹

json_path = r"datasets/lables/val"  # 标注数据json的文件地址
txt_path = r"datasets/lables/val"  # 转换后的txt标签文件存放的文件夹

# json文件名列表
json_name_list = os.listdir(json_path)

# txt文件名列表
txt_name_list = []
for n in range(len(json_name_list)):
    txt_name_list.append(os.path.splitext(json_name_list[n])[0] + '.txt')

# json文件路径列表
json_path_list = []
for foldername in json_name_list:
    json_path_list.append(os.path.join(json_path, foldername))

# txt文件路径列表
txt_path_list = []
for foldername in txt_name_list:
    txt_path_list.append(os.path.join(txt_path, foldername))

# folder_num 个json文件
for folder_num in range(len(json_name_list)):
    # 打开每个json
    with open(json_path_list[folder_num], 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

        # 获取图片的宽、高像素数
        height = json_data['imageHeight']
        width = json_data['imageWidth']

        # 其他需要的数据
        label_number_list = []           # 标签序号
        center_point_width_list = []     # 标签中心点在宽度方向百分比
        center_point_height_list = []    # 标签中心点在高度方向百分比
        percentage_width_list = []       # 标签宽度百分比
        percentage_height_list = []      # 标签高度百分比

        # 标了几个标签？
        label_num = len(json_data['shapes'])
        # 获取每个标签的百分比位置
        for n in range(label_num):
            # 读取第 n 个标签
            # label = json_data['shapes'][n]
            # print(label)
            # {'label': '111',
            #  'points': [[86.65346534653466, 82.82178217821782],
            #             [372.7920792079208, 301.63366336633663]],
            #  'group_id': None,
            #  'description': '',
            #  'shape_type': 'rectangle',
            #  'flags': {},
            #  'mask': None}

            # 标签编号
            label_number_list.append(json_data['shapes'][n]['label'])

            # 标签的两点位置
            point1 = json_data['shapes'][n]['points'][0]  # 点的像素位置[先从左到右，再从上到下]
            point2 = json_data['shapes'][n]['points'][1]

            # 中心点的百分比位置
            center_point_width_list.append(((point1[0] + point2[0]) / 2) / width)
            center_point_height_list.append(((point1[1] + point2[1]) / 2) / height)
            center_point_width_list[n] = round(center_point_width_list[n], 6)
            center_point_height_list[n] = round(center_point_height_list[n], 6)

            # 框的百分比宽高
            percentage_width_list.append(abs(point1[0] - point2[0]) / width)
            percentage_height_list.append(abs(point1[1] - point2[1]) / height)
            percentage_width_list[n] = round(percentage_width_list[n], 6)
            percentage_height_list[n] = round(percentage_height_list[n], 6)

        # print(label_number_list)
        # print(center_point_width_list)
        # print(center_point_height_list)
        # print(percentage_width_list)
        # print(percentage_height_list)

    # 写进txt里
    with open(txt_path_list[folder_num], 'w', encoding='utf-8') as txt_file:
        # 共 nn 个标签，一个标签一行
        for nn in range(len(label_number_list)):
            txt_file.write(str(label_number_list[nn]))
            txt_file.write(' ')
            txt_file.write(str(center_point_width_list[nn]))
            txt_file.write(' ')
            txt_file.write(str(center_point_height_list[nn]))
            txt_file.write(' ')
            txt_file.write(str(percentage_width_list[nn]))
            txt_file.write(' ')
            txt_file.write(str(percentage_height_list[nn]))
            txt_file.write('\n')