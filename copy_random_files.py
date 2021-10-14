import os
import shutil
from shutil import copy2
import random

src_path = "/shared-network/syadav/all_domain_images/"
target_path = "/shared-network/syadav/domain_images_random/"

# Delete if target path exist and create one
if os.path.isdir(target_path) and os.path.exists(target_path):
    shutil.rmtree(target_path)
    os.mkdir(target_path)
else:
    os.mkdir(target_path)


def copy_files(id, src_path, src_files, target_path, target_files):
    """
    Copy image files from one directory to another
    """
    copy2(
        os.path.join(src_path, src_files[id]),
        os.path.join(target_path, src_files[id])
    )


# Select random files
file_id_list = random.sample(range(0, 1725), 100)

# Copy random files
for file_id in file_id_list:
    src_files = os.listdir(src_path)
    target_files = os.listdir(target_path)
    copy_files(file_id, src_path, src_files, target_path, target_files)
