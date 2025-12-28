import os,sys
import shutil
import logging

# Configure logging
logging.basicConfig(filename='find-longly-images.log', level=logging.INFO,
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

def find_image_files(folder_path, image_extensions):
    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(tuple(image_extensions)):
                image_files.append(os.path.join(root, file))
    image_files.sort()
    return image_files

def move_to_error_folder(file_path, error_folder):
    if not os.path.exists(error_folder):
        os.makedirs(error_folder)
    shutil.move(file_path, os.path.join(error_folder, os.path.basename(file_path)))
    relPath =  os.path.relpath(file_path, start='images')
    print(f"Moved image file to error folder: {relPath}")
    logging.info(f"Moved image file to error folder: {file_path}")

def check_labels(image_files, labels_folder, error_folder):
    logging.info(f"image files: total={len(image_files)}")
    delCount=0
    for image_path in image_files:
        # Construct the expected label file path
        relative_path = os.path.relpath(image_path, start='images')
        label_path = os.path.join(labels_folder, relative_path.replace('.jpg', '.txt').replace('.png', '.txt'))

        if not os.path.exists(label_path):
            delCount +=1
            labelRelPath = os.path.relpath(label_path, start='labels')
            logging.info(f"{delCount}-not found:{labelRelPath}, for {relative_path}")
            #print(f"NO{delCount}-:{label_path} not found..")
            move_to_error_folder(image_path, error_folder)
        
        #break

if __name__ == "__main__":
    dataset_folder = 'path/to/dataset'  # Path to the dataset folder
    dataset_folder = '/home/eton/00-src/yolo-ultralytics-250101/datasets/thyNoduBoMSegV01'  # Path to the dataset folder
    images_folder = os.path.join(dataset_folder, 'images')
    labels_folder = os.path.join(dataset_folder, 'labels')
    error_folder = os.path.join(dataset_folder, 'error-longlyImgs')  # Path to the error folder

    image_extensions = ['.jpg', '.png', '.jpeg']  # Add more extensions if needed

    # Find all image files
    image_files = find_image_files(images_folder, image_extensions)

    # Check for corresponding label files
    check_labels(image_files, labels_folder, error_folder)

    logging.info("Image and label validation completed.")

