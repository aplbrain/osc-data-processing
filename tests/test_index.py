from unittest import TestCase
from index import COCOIndex
import pandas as pd
from itertools import combinations


class COCOIndexTestCase(TestCase):
    def setUp(self):
        self.index = COCOIndex('', load=False)
        self.index._index = pd.DataFrame(
            columns=[
                'ImageID', 'NumInstances_1', 'NumInstances_2', 'NumInstances_3', 'NumInstances_4', 'NumInstances_5'
            ],
            data=[
                ['Empty', 0, 0, 0, 0, 0],
                ['Has1', 1, 0, 0, 0, 0],
                ['Has2', 0, 2, 0, 0, 0],
                ['Has3', 0, 0, 3, 0, 0],
                ['Has4', 0, 0, 0, 4, 0],
                ['Has5', 0, 0, 0, 0, 5],
                ['Has12', 1, 2, 0, 0, 0],
                ['Has13', 3, 0, 4, 0, 0],
                ['Has14', 5, 0, 0, 1, 0],
                ['Has15', 2, 0, 0, 0, 3],
                ['Has123', 4, 5, 1, 0, 0],
                ['Has124', 2, 3, 0, 4, 0],
                ['Has135', 5, 0, 1, 0, 2],
                ['Has1235', 3, 4, 5, 0, 1],
                ['Has1345', 2, 0, 3, 4, 5],
                ['Has12345', 1, 2, 3, 4, 5],
            ],
        )

    def test_get_classes(self):
        self.assertSetEqual(self.index.get_classes(), {1, 2, 3, 4, 5})

    def test_get_images(self):
        self.assertSetEqual(self.index.get_images(), {
            'Empty', 'Has1', 'Has2', 'Has3', 'Has4', 'Has5', 'Has12', 'Has13', 'Has14', 'Has15', 'Has123', 'Has124',
            'Has135', 'Has1235', 'Has1345', 'Has12345',
        })

    def test_get_images_with_classes(self):
        # all classes
        self.assertSetEqual(self.index.get_images_with_classes(list(self.index.get_classes())),
                            self.index.get_images() - {'Empty'})

        # all combinations of classes
        for i in range(1, 6):
            for class_ids in combinations(self.index.get_classes(), i):
                self.assertSetEqual(
                    self.index.get_images_with_classes(list(class_ids)),
                    {iid for iid in self.index.get_images() if any(str(cid) in iid for cid in class_ids)}
                )

    def test_get_bounded_num_instances_lower(self):
        self.assertSetEqual(self.index.get_images_with_bounded_num_instances([1], lower=1), {
            'Has1', 'Has12', 'Has13', 'Has14', 'Has15', 'Has123', 'Has124', 'Has135', 'Has1235', 'Has1345', 'Has12345'
        })

        self.assertSetEqual(self.index.get_images_with_bounded_num_instances([1], lower=2), {
            'Has13', 'Has14', 'Has15', 'Has123', 'Has124', 'Has135', 'Has1235', 'Has1345'
        })

        self.assertSetEqual(self.index.get_images_with_bounded_num_instances([1], lower=3), {
            'Has13', 'Has14', 'Has123', 'Has135', 'Has1235'
        })

        self.assertSetEqual(self.index.get_images_with_bounded_num_instances([1], lower=4), {
            'Has14', 'Has123', 'Has135'
        })

        self.assertSetEqual(self.index.get_images_with_bounded_num_instances([1], lower=5), {
            'Has14', 'Has135'
        })

        self.assertEqual(len(self.index.get_images_with_bounded_num_instances([1], lower=6)), 0)

    def test_get_bounded_num_instances_upper(self):
        self.assertSetEqual(self.index.get_images_with_bounded_num_instances([1], upper=6), {
            'Empty', 'Has1', 'Has2', 'Has3', 'Has4', 'Has5', 'Has12', 'Has13', 'Has14', 'Has15', 'Has123', 'Has124',
            'Has135', 'Has1235', 'Has1345', 'Has12345',
        })

        self.assertSetEqual(self.index.get_images_with_bounded_num_instances([1], upper=5), {
            'Empty', 'Has1', 'Has2', 'Has3', 'Has4', 'Has5', 'Has12', 'Has13', 'Has15', 'Has123', 'Has124', 'Has1235',
            'Has1345', 'Has12345',
        })

        self.assertSetEqual(self.index.get_images_with_bounded_num_instances([1], upper=4), {
            'Empty', 'Has1', 'Has2', 'Has3', 'Has4', 'Has5', 'Has12', 'Has13', 'Has15', 'Has124', 'Has1235', 'Has1345',
            'Has12345',
        })

        self.assertSetEqual(self.index.get_images_with_bounded_num_instances([1], upper=3), {
            'Empty', 'Has1', 'Has2', 'Has3', 'Has4', 'Has5', 'Has12', 'Has15', 'Has124', 'Has1345', 'Has12345',
        })

        self.assertSetEqual(self.index.get_images_with_bounded_num_instances([1], upper=2), {
            'Empty', 'Has1', 'Has2', 'Has3', 'Has4', 'Has5', 'Has12', 'Has12345',
        })

        self.assertSetEqual(self.index.get_images_with_bounded_num_instances([1], upper=1), {
            'Empty', 'Has2', 'Has3', 'Has4', 'Has5',
        })

    def test_get_bounded_num_instances_lower_upper(self):
        self.assertSetEqual(self.index.get_images_with_bounded_num_instances([1], lower=2, upper=3), {
            'Has15', 'Has124', 'Has1345',
        })

        self.assertSetEqual(self.index.get_images_with_bounded_num_instances([1], lower=1, upper=3), {
            'Has1', 'Has12', 'Has15', 'Has124', 'Has1345', 'Has12345'
        })

        self.assertSetEqual(self.index.get_images_with_bounded_num_instances([1], lower=2, upper=4), {
            'Has13', 'Has15', 'Has124', 'Has1235', 'Has1345',
        })
