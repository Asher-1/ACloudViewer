# encoding: utf-8

import numpy as np
import cloudViewer as cv3d


def build_vocab_tree(database_path, vocab_tree_path, num_visual_words=65536,
                     num_checks=256, branching=256,
                     num_iterations=11, max_num_images=-1):
    """
    build_vocab_tree(database_path, vocab_tree_path, num_visual_words=65536, num_checks=256,
    branching=256, num_iterations=11, max_num_images=-1, num_threads=-1)
    Function for the building of vocabulary tree
    
    Args:
        database_path (str): Path to database in which to store the extracted data
        vocab_tree_path (str): The vocabulary tree path.
        num_visual_words (int, optional, default=65536): The desired number of visual words,
        i.e. the number of leaf node clusters. Note that the actual number of visual words might be less.
        num_checks (int, optional, default=256): The number of checks in the nearest neighbor search.
        branching (int, optional, default=256): The branching factor of the hierarchical k-means tree.
        num_iterations (int, optional, default=11): The number of iterations for the clustering.
        max_num_images (int, optional, default=-1): The maximum number of images.

    Returns:
        int
    """
    return cv3d.reconstruction.vocab_tree.build_vocab_tree(database_path=database_path, vocab_tree_path=vocab_tree_path,
                                                           num_visual_words=num_visual_words, num_checks=num_checks,
                                                           branching=branching, num_iterations=num_iterations,
                                                           max_num_images=max_num_images)


def retrieve_vocab_tree(database_path, vocab_tree_path, output_index_path='', query_image_list_path='',
                        database_image_list_path='', max_num_images=-1, num_neighbors=5,
                        num_checks=256, num_images_after_verification=0, max_num_features=-1):
    """
    retrieve_vocab_tree(database_path, vocab_tree_path, output_index_path='', query_image_list_path='',
    database_image_list_path='', max_num_images=-1, num_neighbors=5, num_checks=256,
    num_images_after_verification=0, max_num_features=-1, num_threads=-1)
    Function for the retrieve of vocabulary tree
    
    Args:
        database_path (str): Path to database in which to store the extracted data
        vocab_tree_path (str): The vocabulary tree path.
        output_index_path (str, optional, default=''): The output index path.
        query_image_list_path (str, optional, default=''): The query image list path.
        database_image_list_path (str, optional, default=''): The database image list path.
        max_num_images (int, optional, default=-1): The maximum number of images.
        num_neighbors (int, optional, default=5): The number of nearest neighbor visual words
         that each feature descriptor is assigned to.
        num_checks (int, optional, default=256): The number of checks in the nearest neighbor search.
        num_images_after_verification (int, optional, default=0):
        Whether to perform spatial verification after image retrieval.
        max_num_features (int, optional, default=-1): The maximum number of features.

    Returns:
        int
    """
    return cv3d.reconstruction.vocab_tree.retrieve_vocab_tree(database_path=database_path,
                                                              vocab_tree_path=vocab_tree_path,
                                                              output_index_path=output_index_path,
                                                              query_image_list_path=query_image_list_path,
                                                              database_image_list_path=database_image_list_path,
                                                              max_num_images=max_num_images,
                                                              num_neighbors=num_neighbors,
                                                              num_checks=num_checks,
                                                              num_images_after_verification=num_images_after_verification,
                                                              max_num_features=max_num_features)


if __name__ == '__main__':
    np.random.seed(42)
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)

    DATABASE_PATH = "/media/asher/data/datasets/gui_test/database.db"
    OUTPUT_PATH = "/media/asher/data/datasets/gui_test/out/vocab_tree_index.bin"
    VOCAB_TREE_PATH = "/media/asher/data/datasets/vocab_tree/vocab_tree_flickr100K_words32K.bin"

    flag = build_vocab_tree(database_path=DATABASE_PATH, vocab_tree_path=VOCAB_TREE_PATH)
    if flag != 0:
        print("build_vocab_tree failed!")

    flag = retrieve_vocab_tree(database_path=DATABASE_PATH,
                               vocab_tree_path=VOCAB_TREE_PATH,
                               output_index_path=OUTPUT_PATH)
    if flag != 0:
        print("retrieve_vocab_tree failed!")
