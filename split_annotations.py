# Copyright 2020 The Johns Hopkins University Applied Physics Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import argparse
import os
import multiprocessing
from tqdm import tqdm
from annotations import mapillary_annotations, isaid_annotations
from typing import List, Dict, Any


def save_annotations(output_dir: str, filename: str, annotations: List[Dict[str, Any]]):
    """
    Saves a collection of annotations for a single image to a json file.

    :param output_dir: The directory to save the single annotations file in.
    :param filename: The filename of image.
    :param annotations: The list of annotations
    """
    with open(os.path.join(output_dir, os.path.splitext(filename)[0] + '.json'), 'w') as fp:
        json.dump({
            'annotations': annotations
        }, fp)
    return filename


def split_sequentially(output_dir: str, annotations_by_filename: Dict[str, List[Dict[str, Any]]]):
    """
    Splits the annotations sequentially using `save_annotations`.

    Useful for debugging purposes.

    :param output_dir: The directory to save the split annotations to.
    :param annotations_by_filename: The dictionary of annotations. See annotations.py file
    """
    for filename, annotations in tqdm(annotations_by_filename.items()):
        save_annotations(output_dir, filename, annotations)


def wrapped_save_annotations(args):
    return save_annotations(*args)


def split_multiprocessing(output_dir: str, annotations_by_filename: Dict[str, List[Dict[str, Any]]]):
    """
    Splits the annotations using multiple processes with `multiprocessing`. Faster than `split_sequentially`. It calls
    `save_annotations`.

    :param output_dir: The directory to save the split annotations to.
    :param annotations_by_filename: The dictionary of annotations. See annotations.py file.
    """

    with multiprocessing.Pool() as pool:
        jobs = [(output_dir, filename, annotations) for filename, annotations in annotations_by_filename.items()]
        for filename in tqdm(pool.imap_unordered(wrapped_save_annotations, jobs), total=len(jobs)):
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='The path to the single JSON file containing COCO style annotations.')
    parser.add_argument('output_dir', help='The directory to output the annotations to')
    parser.add_argument('--dataset', choices=['mapillary', 'isaid'], help='The dataset the JSON file is for.')
    args = parser.parse_args()

    path = args.path

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == 'mapillary':
        annotations_fn = mapillary_annotations
    elif args.dataset == 'isaid':
        annotations_fn = isaid_annotations
    elif 'mapillary' in path.lower():
        annotations_fn = mapillary_annotations
    elif 'isaid' in path.lower():
        annotations_fn = isaid_annotations
    else:
        raise ValueError('Unknown annotations style')

    categories, annotations_by_filename = annotations_fn(args.path)

    split_multiprocessing(args.output_dir, annotations_by_filename)


if __name__ == '__main__':
    main()
