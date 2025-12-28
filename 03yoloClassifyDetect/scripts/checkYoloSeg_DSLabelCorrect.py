import os,sys
import shutil
import logging
from pathlib import Path 

# Configure logging
logging.basicConfig(filename='segLabel_check.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def confirm_action(prompt):
    while True:
        response = input(f"{prompt} (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'.")

def is_valid_label_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            Nof = len(lines)
            if 1!=Nof:
                return False
            for line in lines:
                parts = line.split()
                if len(parts) < (3*2+1):#3points(x,y) + classIndex=7
                    return False
                class_id =int(parts[0])
                if not (0 <= class_id < 100):
                    logging.error(f"class_id Err:{class_id}")
                    return False
                coordinates = map(float, parts[1:])
                for icord in coordinates:
                    if icord <0 or icord >1:
                        return False
        logging.info(f"Normal:{file_path}")
        return True
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return False

def get_relative_path_to_labels(absolute_path, base_dir_name='labels'):
    # Find the position of the base directory in the absolute path
    parts = Path(absolute_path).parts
    try:
        base_index = parts.index(base_dir_name)
    except ValueError:
        raise ValueError(f"The base directory '{base_dir_name}' is not found in the path.")

    # Construct the base directory path
    base_dir_path = Path(*parts[:base_index + 1])

    # Compute the relative path from the base directory
    relative_path = os.path.relpath(absolute_path, start=base_dir_path)

    return relative_path

def move_to_error_folder(file_path, error_folder):
    if not os.path.exists(error_folder):
        os.makedirs(error_folder)
    shutil.move(file_path, os.path.join(error_folder, os.path.basename(file_path)))
    logging.error(f"Moved invalid label file to error folder: {file_path}")
    
    # move bind image too.
    #print(f"relPath0={file_path}")
    relative_path = get_relative_path_to_labels(file_path)
    parts=file_path.split(os.sep)
    labelsPartIdx = parts.index('labels')
    basePath = os.sep.join(parts[:labelsPartIdx])
    imgbasePath = os.path.join(basePath, 'images')
    #print(f"imagebasepath:{imgbasePath}, relPath={relative_path}")

    imgf_path = os.path.join(imgbasePath, relative_path.replace('.txt', '.jpg').replace('.txt', '.png'))
    shutil.move(imgf_path, os.path.join(error_folder, os.path.basename(imgf_path)))


def check_labels_in_folder(folder_path, error_folder):
    msg=f"process: {folder_path}"
    print(msg)
    filenames = os.listdir(folder_path)
    print(f"filenames len={len(filenames)}")
    for filename in filenames:
        if not filename.endswith('.txt'):
            continue
        file_path = os.path.join(folder_path, filename)
        if not is_valid_label_file(file_path):
            move_to_error_folder(file_path, error_folder)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: app dataset folder")

    labels_folder = os.path.abspath(sys.argv[1])#'/home/eton/00-src/yolo-ultralytics-250101/datasets/thyNoduBoMSegV01/labels'  # Path to the folder containing label files
    
    error_folder = os.path.join(labels_folder, 'err-labels') #'/home/eton/00-src/yolo-ultralytics-250101/datasets/thyNoduBoMSegV01/label-error'  # Path to the folder where invalid labels will be moved
    labels_folder = os.path.join(labels_folder, 'labels')
    if confirm_action(f"process {labels_folder} ?"):
        print("Operation confirmed. Proceeding...")
        subfolders=['train', 'val']
        for ifolder in subfolders:
            check_labels_in_folder(labels_folder+'/'+ifolder, error_folder)
        logging.info("Label validation completed.")
    else:
        print("Operation canceled.")


