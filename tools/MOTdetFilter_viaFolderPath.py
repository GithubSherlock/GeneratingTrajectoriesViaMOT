import os
import numpy as np

input_folder_path = './input/mensa_yolov8s/'
output_folder_path = './output/output_detFileFilter/tracker_car/mensa_yolov8s/'
file_name = 'det_mensa_cls.txt'
class_id = 'car'  # class_id: 'car', 'bicycle', 'truck', 'motorbike', 'bus', even 'person'

# def stat_tra_filter(det_file, class_id):
#     obj_car = []
#     cars_cp = []
#     breakline = []
#     stat_car = []
#     filter_list = []
#
#     for line in det_file:
#         if line[-1] == class_id:
#             cls_id = line[-1]
#             obj_car.append(line)
#             obj_car = sorted(obj_car, key=lambda x: int(x[1]))
#
#     for line in obj_car:
#         cars_cp.append([line[0], line[1], int(line[2]) + int(line[4]) / 2, int(line[3]) + int(line[5]) / 2])
#
#     # 遍历 cars_cp 列表，从第二行开始比较
#     for i in range(1, len(cars_cp)):
#         # 获取当前行和上一行的数据
#         # current_frame, current_track_id, _, _ = cars_cp[i]
#         # previous_frame, previous_track_id, _, _ = cars_cp[i - 1]
#
#         # 比较帧数之差和 track_id 是否相同
#         if int(cars_cp[i][0]) - int(cars_cp[i-1][0]) > 1 or cars_cp[i][1] != cars_cp[i-1][1]:
#             breakline.append(cars_cp[i])  # 将当前行添加到 breakline 列表中
#
#     filtered_cars_cp = cars_cp.copy()
#     for check in range(1, len(breakline)):
#         # 初始化变量，用于计算均值
#         x_sum = 0
#         y_sum = 0
#         w_sum = 0
#         h_sum = 0
#         count = 0  # 用于记录符合条件的数据行数
#         # 遍历 obj_car 列表，找到符合条件的数据行
#         for line in obj_car:
#             track_id = breakline[check][1]
#             # 检查是否为目标 track_id 且 frame_number 在 a 到 b 之间
#             if track_id == line[1] and int(breakline[check-1][0]) <= int(line[0]) <= int(breakline[check][0])-1:
#                 # 累加第二列至第五列的值
#                 x_sum += int(line[2])
#                 y_sum += int(line[3])
#                 w_sum += int(line[4])
#                 h_sum += int(line[5])
#                 count += 1  # 符合条件的数据行数加一
#         # 计算均值
#         x_mean = x_sum / count if count > 0 else 0
#         y_mean = y_sum / count if count > 0 else 0
#         w_mean = w_sum / count if count > 0 else 0
#         h_mean = h_sum / count if count > 0 else 0
#         # 筛选 cars_cp 列表中的数据
#         filtered_row = [car for car in filtered_cars_cp if (int(breakline[check-1][0]) <= int(car[0]) <= int(breakline[check][0])-1 and
#                                                                 car[1] == track_id and
#                                                                 x_mean < float(car[2]) < x_mean + w_mean and
#                                                                 y_mean < float(car[3]) < y_mean + h_mean)]
#         filter_list.append(filtered_row)
#
#     items = [element for sublist in filter_list for element in (sublist if isinstance(sublist, list) else [sublist])]
#     seen = set()
#     for item in items:
#         # 将列表转换为元组
#         t_item = tuple(item)
#         if t_item not in seen:
#             seen.add(t_item)
#             stat_car.append(item)
#     # 创建一个布尔数组，初始时假设所有行都应该保留
#     mask = np.ones(len(det_file), dtype=bool)
#     # 遍历 stat_car 中的每一行
#     for frame_number, track_id, _, _ in stat_car:
#         # 更新 mask，找到与 stat_car 中当前行匹配的行，并将它们标记为 False（不保留）
#         mask &= ~((det_file[:, 0].astype(int) == np.array(frame_number)) &
#                 (det_file[:, 1].astype(int) == np.array(track_id)) &
#                 (det_file[:, -1].astype(str) == np.array(cls_id)))
#     # 应用 mask，删除那些不应保留的行
#     det_file_filtered = det_file[mask]
#
#     return det_file_filtered

def sta_tra_filter(det_file, class_id):
    filter_list = []
    stat_car = []
    # 使用列表推导式过滤出所有属于class_id类别的行，并按照第二个字段即track_id排序
    obj_car = sorted([line for line in det_file if line[-1] == class_id], key=lambda x: int(x[1]))
    obj_car = [line[:-1] for line in obj_car]
    # 将 obj_car 转换为 NumPy 数组以利用向量化操作
    obj_car_np = np.array(obj_car, dtype=int)

    # 直接在列表推导式中完成中心点的计算
    cars_cp = [[int(line[0]), line[1], int(line[2]) + int(line[4]) / 2, int(line[3]) + int(line[5]) / 2] for line in
               obj_car]

    # 使用NumPy数组来处理breakline的逻辑
    cars_cp_np = np.array(cars_cp, dtype=int)
    # 计算帧数之差
    frame_diffs = np.diff(cars_cp_np[:, 0])
    # 检查track_id是否相同
    track_id_changes = np.where(cars_cp_np[1:, 1] != cars_cp_np[:-1, 1])[0]
    # 找到breakline的索引
    breakline_indices = np.where((frame_diffs > 1) | np.in1d(np.arange(len(frame_diffs)), track_id_changes))[0] + 1
    breakline = cars_cp_np[breakline_indices].tolist()

    for check in range(1, len(breakline)):
        track_id = int(breakline[check][1])
        frame_start = int(breakline[check - 1][0])
        frame_end = int(breakline[check][0]) - 1

        # 使用布尔索引选择符合条件的行
        condition = (obj_car_np[:, 1] == track_id) & (obj_car_np[:, 0] >= frame_start) & (obj_car_np[:, 0] <= frame_end)
        selected_rows = obj_car_np[condition]

        # 如果有符合条件的行，计算均值
        if selected_rows.size > 0:
            x_mean, y_mean, w_mean, h_mean = selected_rows[:, 2:6].mean(axis=0)

            # 筛选中心点坐标在均值附近的车辆
            filtered_rows = selected_rows[
                (selected_rows[:, 2] > x_mean) & (selected_rows[:, 2] < x_mean + w_mean) &
                (selected_rows[:, 3] > y_mean) & (selected_rows[:, 3] < y_mean + h_mean)
                ]
            filter_list.extend(filtered_rows.tolist())  # 将 NumPy 数组转换回列表
    filter_list = sorted(filter_list, key=lambda x: (x[0], x[1]))

    return np.array(filter_list)

def frame_filter(lines):
    data = np.array(lines, dtype=int)
    filtered_dataset = data[data[:, 0] % 10 == 0]

    return filtered_dataset


def det_file_filter(input_folder, output_folder):
    for subdir in os.listdir(input_folder):
        subdir_path = os.path.join(input_folder, subdir)
        if os.path.isdir(subdir_path):
            output_subdir_path = os.path.join(output_folder, subdir)
            os.makedirs(output_subdir_path, exist_ok=True)

            for file in os.listdir(subdir_path):
                if file.endswith('.txt'):
                    file_path = os.path.join(subdir_path, file)
                    with open(file_path, 'r') as f:
                        det_file = [line.strip().split(',') for line in f]
                        print("File reading: {}".format(file_path))

                        det_file_filtered = sta_tra_filter(det_file, class_id)
                        print("det_file_filtered!")

                        filtered_dataset = frame_filter(det_file_filtered)  # det_file
                        print("processed_lines!")

                        output_file_path = os.path.join(output_subdir_path, file)
                        with open(output_file_path, 'w') as f_out:
                            for row in filtered_dataset:
                                f_out.write(','.join(map(str, row)) + '\n')  # 将每一行转换为逗号分隔的字符串
                            print("File has been written!")

if __name__ == '__main__':
    det_file_filter(input_folder_path, output_folder_path)
    print("Save successfully!")

