import os
import numpy as np

input_folder_path = './input/mensa_500/'
output_folder_path = './output/output_detFileFilter/tracker_bicycle/mensa_500/'
file_name = 'det_mensa_cls.txt'
class_id = 'bicycle'  # class_id: 'car', 'bicycle', 'truck', 'motorbike', 'bus', even 'person'

def class_filter(lines, class_id):
    # 使用列表推导式优化循环，并且移除空字符串
    processed_lines = [','.join(filter(None, line[:-1])) for line in lines if int(line[0]) % 10 == 0 and
                       line[-1].strip() == class_id]

    return np.array(processed_lines)

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

                        filtered_dataset = class_filter(det_file, class_id)
                        print("processed_lines!")

                        output_file_path = os.path.join(output_subdir_path, file)
                        with open(output_file_path, 'w') as f_out:
                            for row in filtered_dataset:
                                f_out.write(row + '\n')
                            print("File has been written!")

if __name__ == '__main__':
    det_file_filter(input_folder_path, output_folder_path)
    print("Save successfully!")
