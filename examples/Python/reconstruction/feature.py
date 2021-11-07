# CloudViewer: Asher-1.github.io
# The MIT License (MIT)
# See license file or visit Asher-1.github.io for details

# examples/Python/reconstruction/feature.py

import time
import numpy as np
import cloudViewer as cv3d


def import_feature(database_path, image_path,
                   import_path, image_list_path="", camera_mode=0):
    image_reader_options = cv3d.reconstruction.options.ImageReaderOptions()
    if not image_reader_options.check():
        return 1
    sift_extraction_options = cv3d.reconstruction.options.SiftExtractionOptions()
    if not sift_extraction_options.check():
        return 1
    return cv3d.reconstruction.feature.import_feature(database_path=database_path,
                                                      image_path=image_path,
                                                      import_path=import_path,
                                                      image_list_path=image_list_path,
                                                      camera_mode=camera_mode,
                                                      image_reader_options=image_reader_options,
                                                      sift_extraction_options=sift_extraction_options)


def import_matches(database_path, match_list_path="", match_type="pairs"):
    sift_matching_options = cv3d.reconstruction.options.SiftMatchingOptions()
    if not sift_matching_options.check():
        return 1
    return cv3d.reconstruction.feature.import_matches(database_path=database_path,
                                                      match_list_path=match_list_path,
                                                      match_type=match_type,
                                                      sift_matching_options=sift_matching_options)


def extract_features(database_path, image_path, image_list_path="", camera_mode=0):
    image_reader_options = cv3d.reconstruction.options.ImageReaderOptions()
    if not image_reader_options.check():
        return 1
    sift_extraction_options = cv3d.reconstruction.options.SiftExtractionOptions()
    if not sift_extraction_options.check():
        return 1

    return cv3d.reconstruction.feature.extract_feature(database_path=database_path,
                                                       image_path=image_path,
                                                       image_list_path=image_list_path,
                                                       camera_mode=camera_mode,
                                                       image_reader_options=image_reader_options,
                                                       sift_extraction_options=sift_extraction_options)


def exhaustive_match(database_path):
    sift_matching_options = cv3d.reconstruction.options.SiftMatchingOptions()
    if not sift_matching_options.check():
        return 1
    exhaustive_matching_options = cv3d.reconstruction.options.ExhaustiveMatchingOptions()
    if not exhaustive_matching_options.check():
        return 1
    return cv3d.reconstruction.feature.exhaustive_match(database_path=database_path,
                                                        sift_matching_options=sift_matching_options,
                                                        exhaustive_matching_options=exhaustive_matching_options)


def sequential_match(database_path):
    sift_matching_options = cv3d.reconstruction.options.SiftMatchingOptions()
    if not sift_matching_options.check():
        return 1
    sequential_matching_options = cv3d.reconstruction.options.SequentialMatchingOptions()
    if not sequential_matching_options.check():
        return 1
    return cv3d.reconstruction.feature.sequential_match(database_path=database_path,
                                                        sift_matching_options=sift_matching_options,
                                                        sequential_matching_options=sequential_matching_options)


def spatial_match(database_path):
    sift_matching_options = cv3d.reconstruction.options.SiftMatchingOptions()
    if not sift_matching_options.check():
        return 1
    spatial_matching_options = cv3d.reconstruction.options.SpatialMatchingOptions()
    if not spatial_matching_options.check():
        return 1
    return cv3d.reconstruction.feature.spatial_match(database_path=database_path,
                                                     sift_matching_options=sift_matching_options,
                                                     spatial_matching_options=spatial_matching_options)


def transitive_match(database_path):
    sift_matching_options = cv3d.reconstruction.options.SiftMatchingOptions()
    if not sift_matching_options.check():
        return 1
    transitive_matching_options = cv3d.reconstruction.options.TransitiveMatchingOptions()
    if not transitive_matching_options.check():
        return 1
    return cv3d.reconstruction.feature.transitive_match(database_path=database_path,
                                                        sift_matching_options=sift_matching_options,
                                                        transitive_matching_options=transitive_matching_options)


def vocab_tree_match(database_path, vocab_tree_path):
    sift_matching_options = cv3d.reconstruction.options.SiftMatchingOptions()
    if not sift_matching_options.check():
        return 1
    vocab_tree_matching_options = cv3d.reconstruction.options.VocabTreeMatchingOptions()
    vocab_tree_matching_options.vocab_tree_path = vocab_tree_path
    if not vocab_tree_matching_options.check():
        return 1
    return cv3d.reconstruction.feature.vocab_tree_match(database_path=database_path,
                                                        sift_matching_options=sift_matching_options,
                                                        vocab_tree_matching_options=vocab_tree_matching_options)


if __name__ == "__main__":
    np.random.seed(42)
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)

    DATABASE_PATH = "/media/asher/data/datasets/out/sql.db"
    IMAGE_PATH = "/media/asher/data/datasets/dataset_monstree/mini6"
    VOCAB_TREE_PATH = "/media/asher/data/datasets/vocab_tree/vocab_tree_flickr100K_words32K.bin"
    IMPORT_PATH = ""
    IMAGE_LIST_PATH = ""
    MATCH_LIST_PATH = ""
    MATCH_TYPE = "pairs"
    CAMERA_MODE = 0

    # flag = import_feature(DATABASE_PATH, IMAGE_PATH, IMPORT_PATH, IMAGE_LIST_PATH, CAMERA_MODE)
    # if flag != 0:
    #     print("import_feature failed!")

    # flag = import_matches(DATABASE_PATH, MATCH_LIST_PATH, MATCH_TYPE)
    # if flag != 0:
    #     print("import_feature failed!")

    flag = extract_features(DATABASE_PATH, IMAGE_PATH, IMAGE_LIST_PATH, CAMERA_MODE)
    if flag != 0:
        print("extract_feature failed!")

    flag = exhaustive_match(DATABASE_PATH)
    if flag != 0:
        print("exhaustive_match failed!")

    flag = sequential_match(DATABASE_PATH)
    if flag != 0:
        print("sequential_match failed!")

    flag = spatial_match(DATABASE_PATH)
    if flag != 0:
        print("spatial_match failed!")

    flag = transitive_match(DATABASE_PATH)
    if flag != 0:
        print("transitive_match failed!")

    flag = vocab_tree_match(DATABASE_PATH, VOCAB_TREE_PATH)
    if flag != 0:
        print("vocab_tree_match failed!")
