import os

APP_FOLDER = '../data/images'
totalFiles = 0
for base, dirs, files in os.walk(APP_FOLDER):
    for Files in files:
        totalFiles += 1

print('Total number of files', totalFiles)
