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

from PIL import Image
import numpy as np
import json
from typing import Optional, List, Dict, Any


class COCOImage(object):
    """
    An object for loading image and annotations data for a COCO annotated image.

    Example Usage: ::

        # create the object
        coco_image = COCOImage('path/to/image.png', 'path/to/annotations.json')

        # access the actual image data as a numpy array
        coco_image.image

        # print the mean x coordinate for all instances in this image
        print(coco_image.bboxes[:, 0].mean())

        # print the mean area of all the instances in this image
        print(coco_image.areas.mean())
    """

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        return np.asarray(Image.open(image_path))

    @staticmethod
    def load_annotations(annotations_path: str) -> List[Dict[str, Any]]:
        with open(annotations_path) as fp:
            return json.load(fp)['annotations']

    def __init__(self, image_path: str, annotations_path: str, load_image=True, load_annotations=True):
        self.image_path = image_path
        self.annotations_path = annotations_path

        if load_image:
            self.image = self.load_image(image_path)
        else:
            self.image = np.array([])

        if load_annotations:
            self.annotations = self.load_annotations(annotations_path)
        else:
            self.annotations = []

        # a [N, 4] numpy array consisting of [x, y, w, h] coordinates for each instance's bounding boxes
        self.bboxes: np.ndarray = np.array([a['bbox'] for a in self.annotations])

        # a [N,] numpy array consisting of the category ids of each instance
        self.category_ids: np.ndarray = np.array([a['category_id'] for a in self.annotations])

        # a [N,] numpy array consisting of the areas of each instance
        self.areas: np.ndarray = np.array([a['area'] for a in self.annotations])

        # a [N,] numpy array consisting of 0 or 1 indicating whether the instance is part of a crowd
        self.is_crowds: np.ndarray = np.array([a['iscrowd'] for a in self.annotations])
