import os
import glob

# 假设帧图像的宽和高作为输入变量
frame_width = 1920
frame_height = 1080

# 输入文件夹路径
input_folder_path = './output/output_sportscheck_street/dataset_yolo+id/'  # 替换为你的txt文件夹路径
output_file_path = os.path.join(input_folder_path, 'gt.txt')

if __name__ == '__main__':
    # 用于存储所有行的列表
    lines_to_write = []

    # 获取文件夹中所有txt文件的路径
    txt_files = sorted(glob.glob(os.path.join(input_folder_path, '*.txt')))

    # 遍历所有txt文件
    for txt_file in txt_files:
        # 从文件名中提取帧编号
        frame_number = int(os.path.basename(txt_file).split('.')[0])

        # 打开并读取每个txt文件
        with open(txt_file, 'r') as file:
            for line in file:
                # 解析每一行数据
                cls_id, track_id, x_center, y_center, w, h = map(float, line.strip().split(' '))
                cls_id = int(cls_id)
                track_id = int(track_id)

                # 只处理cls_id为1的情况，或者替换成你想要的类别
                if cls_id != 15: # "person"： 1， "car": 4, "truck": 9, "motorcycle": 10, "bicycle": 15
                    continue  # 如果cls_id不是1，跳过当前行

                # 计算MOT格式的边界框参数
                bb_left = int(round(x_center * frame_width - (w * frame_width / 2)))
                bb_top = int(round(y_center * frame_height - (h * frame_height / 2)))
                bb_width = int(round(w * frame_width))
                bb_height = int(round(h * frame_height))

                # 构建MOT格式的行
                mot_line = f"{frame_number},{track_id},{bb_left},{bb_top},{bb_width},{bb_height},1,-1,-1,-1\n"

                # 将构建好的行添加到列表中
                lines_to_write.append(mot_line)

    # 根据 frame_number 和 track_id 对行进行排序，确保先转换为整数再排序
    lines_to_write.sort(key=lambda x: (int(x.split(',')[0]), int(x.split(',')[1])))

    # 将所有行写入输出文件
    with open(output_file_path, 'w') as out_file:
        out_file.writelines(lines_to_write)
    print("Conversion completed.")
