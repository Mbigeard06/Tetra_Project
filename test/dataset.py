from pathlib import Path

def count_files(path, nb):
    path = Path(path)
    assert sum(1 for f in path.iterdir() if f.is_file()) == nb


#Initial number of images before spliting
count_files("../dataset/images", 2516)