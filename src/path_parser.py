from os import path

def get_file_path(dir: str, file_name: str) -> str:
    '''
    Method returns path to the file based on the directory and file name

    Parameters:
    dir       - directory in which the file is stored
    file_name - name of the file
    '''
    return path.join(path.abspath('.'), dir, file_name)