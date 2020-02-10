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
from collections import defaultdict
from typing import Tuple, List, Dict, Any


def isaid_annotations(path) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    iSAID stores annotations in a flat array, not grouped by image, so this function reorganizes them.

    iSAID annotations have this data:
        {
        'id': 0,
        'image_id': 0,
        'segmentation': [...],
        'category_id': 1,
        'category_name': 'storage_tank',
        'iscrowd': 0,
        'area': 2580,
        'bbox': [244.0, 1602.0, 62.0, 51.0]
        }

    :param path: the path to the JSON annotations file.
    :return: list of categories, dictionary of filename keys and list of annotations values
    """
    with open(path) as fp:
        obj = json.load(fp)

    filename_by_image_id = {}
    for i in obj['images']:
        filename_by_image_id[i['id']] = i['file_name']

    annotations_by_filename = defaultdict(list)
    for a in obj['annotations']:
        filename = filename_by_image_id[a['image_id']]
        annotations_by_filename[filename].append(a)

    return obj['categories'], dict(annotations_by_filename)


def mapillary_annotations(path) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Mapillary stores annotations already grouped by image, but there is an extra level that this function removes.

    Mapillary annotations have this data:
        {
        'area': 52387,
        'category_id': 3,
        'iscrowd': 0,
        'id': 4409415,
        'bbox': [0, 1838, 3264, 525]
        }

    :param path: the path to the JSON annotations file.
    :return: list of categories, dictionary of filename keys and list of annotations values
    """
    with open(path) as fp:
        obj = json.load(fp)

    annotations_by_filename = {}
    for a in obj['annotations']:
        annotations_by_filename[a['file_name']] = a['segments_info']

    return obj['categories'], annotations_by_filename
