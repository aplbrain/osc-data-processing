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

from typing import List, Set, Optional
import pandas as pd

ImageId = str
ClassId = int


class COCOIndex:
    """
    Index into COCO dataset images.

    Example Usage: ::

        index = COCOIndex('/path/to/index.csv')

        # get all the images *remaining* in the index
        all_images = index.get_images()

        # get all the classes in the index
        all_classes = index.get_classes()

        # get images that have at least 1 instance of one of classes 1, 3, or 5 in them
        image_ids_1 = index.get_images_with_classes([1, 3, 5])

        # get images that have (1 <= {num instances of class 1} < 4)
        image_ids_2 = index.get_images_with_bounded_num_instances([1], lower=1, upper=4)

        # remove specific images
        index.remove(image_ids_1)

        # keep only specified images
        index.keep(image_ids_2)
    """

    def __init__(self, path: str, load=True):
        self._path = path

        self._index = None
        if load:
            self._index = self.load()

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path)

    def _true_selector(self) -> pd.Series:
        return self._index.ImageID != ''

    def _false_selector(self) -> pd.Series:
        return self._index.ImageID == ''

    def get_images(self) -> Set[ImageId]:
        """
        Gets the set of ImageIds that are still remaining in the index.
        """
        return set(self._index.ImageID)

    def get_classes(self) -> Set[ClassId]:
        """
        Gets the set of ClassIds that are in the index.
        """
        return set(int(c.split('_')[1]) for c in self._index.columns if 'NumInstances_' in c)

    def get_images_with_classes(self, classes: List[ClassId]) -> Set[ImageId]:
        """
        Gets all images with at least 1 class from the `class_ids` parameter.
        """
        selector = self._false_selector()

        for class_id in classes:
            selector |= self._index[f'NumInstances_{class_id}'] > 0

        return set(self._index[selector].ImageID)

    def get_images_with_bounded_num_instances(
            self, classes: List[ClassId], lower: Optional[int] = None, upper: Optional[int] = None
    ) -> Set[ImageId]:
        """
        Gets all images where the num instances of each class in `classes` is between `lower` and `upper`:

            lower <= NumInstances_{class_id} < upper
        """
        selector = self._true_selector()

        for class_id in classes:
            if lower is not None:
                selector &= (self._index[f'NumInstances_{class_id}'] >= lower)
            if upper is not None:
                selector &= (self._index[f'NumInstances_{class_id}'] < upper)

        return set(self._index[selector].ImageID)

    def keep(self, image_ids: Set[ImageId]):
        """
        Keep only the image ids specified in the parameter.
        """
        self._index = self._index[self._index.ImageID.isin(image_ids)]

    def remove(self, image_ids: Set[ImageId]):
        """
        Remove only the image ids specified in the parameter.
        """
        self._index = self._index[~self._index.ImageID.isin(image_ids)]
