import os


def get_files_fullpath(dir):
    if dir[-1] != '/': dir += '/'
    files = []
    for entry in os.scandir(dir):
        if entry.is_file():
            files.append('{}{}'.format(dir, entry.name))
    return files
