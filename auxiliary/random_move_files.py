import os
import random
import shutil

source = '../data/images'
dest = '../data/validate/images'

files = os.listdir(source)
no_of_files = 0

for file_name in random.sample(files, no_of_files):
    shutil.move(os.path.join(source, file_name), dest)

# train 147
# test 31
# validate 32
