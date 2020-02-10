# MICrONS Phase 3 data-processing

## Installing Dependencies

```
pip install -r requirements.txt
```

## Splitting COCO annotations

COCO annotations are provided in a single JSON file by default, but you can use
the split_annotations.py script to split it up into a single JSON file for each
image.

Usage:

```
python split_annotations.py <path to COCO annotations> <path to output directory>
python split_annotations.py <path to iSAID>/train/Annotations/iSAID_train.json iSAID_split_annotations/
python split_annotations.py <path to mapillary>/training/panoptic/panoptic_2018.json mapillary_split_annotations/
```

## Creating iSAID Index

1. Download iSAID from: https://captain-whu.github.io/iSAID/dataset.html
2. Create the index with the command:

```
python create_index.py <path to COCO annotations> <path to output file>
```

For example:

```
python create_index.py <path to iSAID>/train/Annotations/iSAID_train.json iSAID_train_index.csv
```

## Creating Mapillary Index

1. Download Mapillary from: https://www.mapillary.com/dataset/vistas
2. Create the index with the command:

```
python create_index.py <path to COCO annotations> <path to output file>
```

For example:

```
python create_index.py <path to mapillary>/training/panoptic/panoptic_2018.json mapillary_training_index.csv
```

## Querying the Index

Use the `COCOIndex` class defined in index.py to query images from an index.

For example:

```
index = COCOIndex('/path/to/index.csv')

# get images that have at least 1 instance of one of classes 1, 3, or 5 in them
image_ids_1 = index.get_images_with_classes([1, 3, 5])

# remove specific images
index.remove(image_ids_1)

# use the remaining images somehow...
index.get_images()
```
