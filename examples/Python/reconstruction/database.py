# CloudViewer: Asher-1.github.io
# The MIT License (MIT)
# See license file or visit Asher-1.github.io for details

# examples/Python/reconstruction/feature.py

import numpy as np
import cloudViewer as cv3d


def merge_database(database_path1, database_path2, out_database_path):
    return cv3d.reconstruction.database.merge_database(database_path1,
                                                       database_path2,
                                                       out_database_path)


def clean_database(database_path, clean_type):
    # supported type {all, images, features, matches}
    return cv3d.reconstruction.database.clean_database(database_path, clean_type)


def create_database(database_path):
    return cv3d.reconstruction.database.create_database(database_path)


if __name__ == "__main__":
    np.random.seed(42)
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)

    DATABASE_PATH = "/media/asher/data/datasets/out/sql.db"
    TYPE = "all"

    flag = clean_database(DATABASE_PATH, TYPE)
    if flag != 0:
        print("clean_database failed!")

    flag = create_database(DATABASE_PATH)
    if flag != 0:
        print("create_database failed!")

    # DATABASE_PATH2 = ""
    # OUT_DATABASE_PATH = ""
    # flag = merge_database(DATABASE_PATH, DATABASE_PATH2, OUT_DATABASE_PATH)
    # if flag != 0:
    #     print("clean_database failed!")
