import os

main_directory = "/securedstorage/DATAsec/cole/Data-secured/opto_benz_1/"

for root, dirs, files in os.walk(main_directory):
    for file in files:
        if file.endswith("_merged.csv"):
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
