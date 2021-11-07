# CloudViewer: Asher-1.github.io
# The MIT License (MIT)
# See license file or visit Asher-1.github.io for details

# examples/Python/reconstruction/feature.py

import numpy as np
import cloudViewer as cv3d


def generate_project(output_path, quality):
    return cv3d.reconstruction.gui.generate_project(output_path=output_path,
                                                    quality=quality)


def gui(database_path, image_path, import_path):
    return cv3d.reconstruction.gui.run_graphical_gui(database_path=database_path,
                                                     image_path=image_path,
                                                     import_path=import_path)


if __name__ == "__main__":
    np.random.seed(42)
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)

    DATABASE_PATH = "/media/asher/data/datasets/gui_test/database.db"
    IMAGE_PATH = "/media/asher/data/datasets/dataset_monstree/mini6"
    IMPORT_PATH = "/media/asher/data/datasets/gui_test/sparse/0"

    OUTPUT_PATH = "/media/asher/data/datasets/gui_test/project.ini"
    QUALITY = "medium"  # {low, medium, high, extreme}

    flag = generate_project(OUTPUT_PATH, QUALITY)
    if flag != 0:
        print("generate_project failed!")

    flag = gui(DATABASE_PATH, IMAGE_PATH, IMPORT_PATH)
    if flag != 0:
        print("gui failed!")
