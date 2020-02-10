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

import os
import argparse
import multiprocessing
from tqdm import tqdm
from annotations import mapillary_annotations, isaid_annotations
from collections import defaultdict
import numpy as np
from typing import List, Dict, Any

"""
Columns:

#. ImageID - id of the image

#. NumInstances - total number of instances in the image
#. NumCategories - total number of categories present in the image

#. TotalArea - sum of instance area over all instances
#. MinArea - min of instance area over all instances
#. MeanArea - mean of instance area over all instances
#. MaxArea - max of instance area over all instances
#. StdevArea - stdev of instance area over all instances
#. MeanX - mean X coordinate over all instances
#. MeanY - mean Y coordinate over all instances
#. StdevX - stdev of X coordinate over all instances
#. StdevY - stdev of Y coordinate over all instances

#. Highest_NumInstances - the highest (num instances in category) over all categories
#. Highest_TotalArea - the highest (total area of instances in category) over all categories
#. Highest_MinArea - the highest (area of smallest instance in category) over all categories
#. Highest_MeanArea - the highest (mean area of instances in category) over all categories
#. Highest_MaxArea - the highest (area of biggest instance in category) over all categories
#. Highest_StdevArea - the highest (stdev of areas of instances in category) over all categories

#. Lowest_NumInstances - the lowest (num instances in category) over all categories
#. Lowest_TotalArea - the lowest (total area of instances in category) over all categories
#. Lowest_MinArea - the lowest (area of smallest instance in category) over all categories
#. Lowest_MeanArea - the lowest (mean area of instances in category) over all categories
#. Lowest_MaxArea - the lowest (area of biggest instance in category) over all categories
#. Lowest_StdevArea - the lowest (stdev of areas of instances in category) over all categories

#. CategoryOf_Highest_NumInstances - the category with the highest num instances
#. CategoryOf_Highest_TotalArea - the category with the highest total area
#. CategoryOf_Highest_MinArea - the category with the largest smallest instance
#. CategoryOf_Highest_MeanArea - the category with the biggest instances on average
#. CategoryOf_Highest_MaxArea - the category with the largest largest instance
#. CategoryOf_Highest_StdevArea - the category with the highest stdev of instance areas

#. CategoryOf_Lowest_NumInstances - the category with the lowest non-zero num instances
#. CategoryOf_Lowest_TotalArea - the category with the lowest non-zero total area
#. CategoryOf_Lowest_MinArea - the category with the smallest smallest instance
#. CategoryOf_Lowest_MeanArea - the category with the smallest instances on average
#. CategoryOf_Lowest_MaxArea - the category with the smallest biggest instance
#. CategoryOf_Lowest_StdevArea - the category with the lowest stdev of instance areas

#. NumInstances_<category id> - the number of instances in <category id>
#. TotalArea_<category id> - the total area of all instances in <category id>
#. MinArea_<category id> - the area of the smallest instance in <category id>
#. MeanArea_<category id> - the mean area of instances in <category id>
#. MaxArea_<category id> - the area of the biggest instance in <category id>
#. StdevArea_<category id> - the stdev of areas of instances in <category id>
#. MeanX_<category id> - the mean X coordinate of instances in <category id>
#. MeanY_<category id> - the mean Y coordinate of instances in <category id>
#. StdevX_<category id> - the stdev of X coordinates of instances in <category id>
#. StdevY_<category id> - the stdev of Y coordinates of instances in <category id>
"""


def create_header_row(category_ids, sep=',') -> str:
    """
    Creates the header row for the index.

    :param category_ids: All of the category IDs.
    :param sep: Optional separator for each column. Default is ','.
    :return: header row string
    """
    columns = [
        'ImageID',
        'NumInstances',
        'NumCategories',
        'TotalArea',
        'MinArea',
        'MeanArea',
        'MaxArea',
        'StdevArea',
        'MeanX',
        'MeanY',
        'StdevX',
        'StdevY',
        'Highest_NumInstances',
        'Highest_TotalArea',
        'Highest_MinArea',
        'Highest_MeanArea',
        'Highest_MaxArea',
        'Highest_StdevArea',
        'Lowest_NumInstances',
        'Lowest_TotalArea',
        'Lowest_MinArea',
        'Lowest_MeanArea',
        'Lowest_MaxArea',
        'Lowest_StdevArea',
        'CategoryOf_Highest_NumInstances',
        'CategoryOf_Highest_TotalArea',
        'CategoryOf_Highest_MinArea',
        'CategoryOf_Highest_MeanArea',
        'CategoryOf_Highest_MaxArea',
        'CategoryOf_Highest_StdevArea',
        'CategoryOf_Lowest_NumInstances',
        'CategoryOf_Lowest_TotalArea',
        'CategoryOf_Lowest_MinArea',
        'CategoryOf_Lowest_MeanArea',
        'CategoryOf_Lowest_MaxArea',
        'CategoryOf_Lowest_StdevArea',
    ]
    for category_id in category_ids:
        columns.extend([
            f'NumInstances_{category_id}',
            f'TotalArea_{category_id}',
            f'MinArea_{category_id}',
            f'MeanArea_{category_id}',
            f'MaxArea_{category_id}',
            f'StdevArea_{category_id}',
            f'MeanX_{category_id}',
            f'MeanY_{category_id}',
            f'StdevX_{category_id}',
            f'StdevY_{category_id}',
        ])

    return sep.join(columns)


def create_index_row(category_ids, filename, annotations, sep=',') -> str:
    """
    Creates an index row for a single image.

    :param category_ids: All of the category IDs.
    :param filename: The image's filename with the extension.
    :param annotations: The list of annotations for this image.
    :param sep: Optional separator for each column. Default is ','.
    :return: string for this image's index row.
    """
    areas = np.array([a['area'] for a in annotations])
    bboxes = np.array([a['bbox'] for a in annotations])

    # group annotations by category_id
    annotations_by_cid = defaultdict(list)
    for a in annotations:
        annotations_by_cid[a['category_id']].append(a)

    # get the areas grouped by category_id
    areas_by_cid = {cid: np.array([a['area'] for a in annotations_by_cid[cid]]) for cid in category_ids}
    bboxes_by_cid = {cid: np.array([a['bbox'] for a in annotations_by_cid[cid]]) for cid in category_ids}

    nonempty_category_ids = [cid for cid in category_ids if len(annotations_by_cid[cid]) > 0]

    # start building the index entry
    row = [
        os.path.splitext(filename)[0],  # ImageID

        len(annotations),  # NumInstances
        len(nonempty_category_ids),  # NumCategories

        # Areas
        areas.sum(),  # TotalArea
        areas.min(),  # MinArea
        areas.mean(),  # MeanArea
        areas.max(),  # MaxArea
        areas.std(),  # StdevArea

        # Coords
        bboxes[:, 0].mean(),  # MeanX
        bboxes[:, 1].mean(),  # MeanY
        bboxes[:, 0].std(),  # StdevX
        bboxes[:, 1].std(),  # StdevY

        # Highest
        max(len(annotations_by_cid[cid]) for cid in nonempty_category_ids),  # Highest_NumInstances
        max(areas_by_cid[cid].sum() for cid in nonempty_category_ids),  # Highest_TotalArea
        max(areas_by_cid[cid].min() for cid in nonempty_category_ids),  # Highest_MinArea
        max(areas_by_cid[cid].mean() for cid in nonempty_category_ids),  # Highest_MeanArea
        max(areas_by_cid[cid].max() for cid in nonempty_category_ids),  # Highest_MaxArea
        max(areas_by_cid[cid].std() for cid in nonempty_category_ids),  # Highest_StdevArea

        # Lowest
        min(len(annotations_by_cid[cid]) for cid in nonempty_category_ids),  # Lowest_NumInstances
        min(areas_by_cid[cid].sum() for cid in nonempty_category_ids),  # Lowest_TotalArea
        min(areas_by_cid[cid].min() for cid in nonempty_category_ids),  # Lowest_MinArea
        min(areas_by_cid[cid].mean() for cid in nonempty_category_ids),  # Lowest_MeanArea
        min(areas_by_cid[cid].max() for cid in nonempty_category_ids),  # Lowest_MaxArea
        min(areas_by_cid[cid].std() for cid in nonempty_category_ids),  # Lowest_StdevArea

        # CategoryOf_Highest
        max(nonempty_category_ids, key=lambda cid: len(annotations_by_cid[cid])),  # CategoryOf_Highest_NumInstances
        max(nonempty_category_ids, key=lambda cid: areas_by_cid[cid].sum()),  # CategoryOf_Highest_TotalArea
        max(nonempty_category_ids, key=lambda cid: areas_by_cid[cid].min()),  # CategoryOf_Highest_MinArea
        max(nonempty_category_ids, key=lambda cid: areas_by_cid[cid].mean()),  # CategoryOf_Highest_MeanArea
        max(nonempty_category_ids, key=lambda cid: areas_by_cid[cid].max()),  # CategoryOf_Highest_MaxArea
        max(nonempty_category_ids, key=lambda cid: areas_by_cid[cid].std()),  # CategoryOf_Highest_StdevArea

        # CategoryOf_Lowest
        min(nonempty_category_ids, key=lambda cid: len(annotations_by_cid[cid])),  # CategoryOf_Lowest_NumInstances
        min(nonempty_category_ids, key=lambda cid: areas_by_cid[cid].sum()),  # CategoryOf_Lowest_TotalArea
        min(nonempty_category_ids, key=lambda cid: areas_by_cid[cid].min()),  # CategoryOf_Lowest_MinArea
        min(nonempty_category_ids, key=lambda cid: areas_by_cid[cid].mean()),  # CategoryOf_Lowest_MeanArea
        min(nonempty_category_ids, key=lambda cid: areas_by_cid[cid].max()),  # CategoryOf_Lowest_MaxArea
        min(nonempty_category_ids, key=lambda cid: areas_by_cid[cid].std()),  # CategoryOf_Lowest_StdevArea
    ]

    for cid in category_ids:
        category_annotations = annotations_by_cid[cid]
        category_areas = areas_by_cid[cid]
        category_bboxes = bboxes_by_cid[cid]

        row.append(len(category_annotations))  # NumInstances_<category id>
        row.append(category_areas.sum() if len(category_annotations) > 0 else 0)  # TotalArea_<category id>
        row.append(category_areas.min() if len(category_annotations) > 0 else 0)  # MinArea_<category id>
        row.append(category_areas.mean() if len(category_annotations) > 0 else 0)  # MeanArea_<category id>
        row.append(category_areas.max() if len(category_annotations) > 0 else 0)  # MaxArea_<category id>
        row.append(category_areas.std() if len(category_annotations) > 0 else 0)  # StdevArea_<category id>
        row.append(category_bboxes[:, 0].mean() if len(category_annotations) > 0 else 0)  # MeanX_<category id>
        row.append(category_bboxes[:, 1].mean() if len(category_annotations) > 0 else 0)  # MeanY_<category id>
        row.append(category_bboxes[:, 0].std() if len(category_annotations) > 0 else 0)  # StdevX_<category id>
        row.append(category_bboxes[:, 1].std() if len(category_annotations) > 0 else 0)  # StdevY_<category id>

    return sep.join(map(str, row))


def create_index_sequentially(output_path: str,
                              categories: List[Dict[str, Any]],
                              annotations_by_filename: Dict[str, List[Dict[str, Any]]],
                              sep: str = ','):
    """
    Creates the index sequentially by calling `create_index_row` on all the images.

    Useful for debugging purposes.

    :param output_path: The path to write the index to.
    :param categories: All of the categories. A list of {'id': ..., 'name': ...} objects.
    :param annotations_by_filename: Annotations grouped by filename. See the annotations.py file.
    :param sep: Optional column separator. Default is ','.
    """
    category_ids = sorted([c['id'] for c in categories])

    with open(output_path, 'w') as fp:
        fp.write(create_header_row(category_ids, sep=sep))
        fp.write('\n')

        for filename, annotations in tqdm(annotations_by_filename.items()):
            row = create_index_row(category_ids, filename, annotations, sep=sep)
            fp.write(row)
            fp.write('\n')


def wrapped_create_index_row(args):
    return create_index_row(*args)


def create_index_multiprocessing(output_path: str,
                                 categories: List[Dict[str, Any]],
                                 annotations_by_filename: Dict[str, List[Dict[str, Any]]],
                                 sep: str = ','):
    """
    Creates the index using multiple processes with `multiprocessing`. It calls `create_index_row` on all the images.

    :param output_path: The path to write the index to.
    :param categories: All of the categories. A list of {'id': ..., 'name': ...} objects.
    :param annotations_by_filename: Annotations grouped by filename. See the annotations.py file.
    :param sep: Optional column separator. Default is ','.
    """

    category_ids = sorted([c['id'] for c in categories])

    with open(output_path, 'w') as fp:
        fp.write(create_header_row(category_ids, sep=sep))
        fp.write('\n')

        with multiprocessing.Pool() as pool:
            jobs = [(category_ids, filename, annotations, sep)
                    for filename, annotations in annotations_by_filename.items()]
            for row in tqdm(pool.imap(wrapped_create_index_row, jobs), total=len(jobs)):
                fp.write(row)
                fp.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='The path to the single JSON file containing COCO style annotations.')
    parser.add_argument('output_path', help='The path to output the index to.')
    parser.add_argument('--dataset', choices=['mapillary', 'isaid'], help='The dataset the JSON file is for.')
    parser.add_argument('--sep', default=',', help='The separator to use for rows. Default is `,`.')
    args = parser.parse_args()

    path = args.path

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

    create_index_multiprocessing(args.output_path, categories, annotations_by_filename, sep=args.sep)


if __name__ == '__main__':
    main()
