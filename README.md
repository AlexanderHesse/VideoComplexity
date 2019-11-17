## About

This is a tool to measure the complexity of an image or video by various methods. Currently, the following methods are available:

Method Name | Description
--- | ---
image:box | Blurs an image using a box filter and returns the absolute difference
image:iterated_box | Applies image:box repeatedly, each time quartering the resolution and quartering the weight of the result
image:gaussian | Blurs an image using a gaussian filter and returns the absolute difference
image:iterated_gaussian | Applies image:gaussian repeatedly, each time quartering the resolution and quartering the weight of the result
image:fft | Reduces the frequencies of the image linearily and returns the sum of all pixels
image:webp | Returns the file size of the image converted to the webp format
video:box | Like image:box, but applies the kernel to the time axis as well
video:gaussian | Like image:gaussian, but applies the kernel to the time axis as well

The results are written in CSV format on stdout.

Since each method has its own scale of complexity, a rescale tool is provided to normalize the results. Currently, the following methods are available:

Parameter | Arguments | Description
--- | --- | ---
--geometric-average | [--reference-file=INPUT_CSV] | scale by the geometric average of a reference result file
--first-frame | [--base-value=1] | scale by normalizing the first frame of each video to a base value

## Requirements

* Python 2
* Numpy
* OpenCV
* Scipy

## Usage

```
video_complexity.py [-v] [-m/--methods <methods>] [-o <out file>] <media>+
rescale.py [-v] <--geometric-average | --first-frame> [--reference-file <reference file>] [--base-value <base value>] [-o <out file>] [csv file]
```

Methods are specified comma-separated and without spaces.

## Examples

```
$ video_complexity.py -v reference_videos/*.mp4 -o reference.csv
$ video_complexity.py -v ducks.mp4 -o ducks.csv
$ rescale.py --reference-file reference.csv --geometric-average ducks.csv -o ducks_normalized.csv
```

```
$ video_complexity.py -m image:gaussian,image:webp,video:gaussian -o ducks.csv ducks.mp4
$ rescale.py --base-value 1000 --first-frame -o ducks_normalized.csv ducks.csv
```

```
$ video_complexity.py -v ducks.mp4 | rescale.py --first-frame > ducks.csv
```

## Implementation Details

* Image methods expect a single image argument, returning the measured complexity.
* Video methods expect a single generator argument. The generator yields a tuple of the frame index and the frame itself. The video method is a generator itself, yielding a tuple of the frame index and the measured complexity.
* Each image or frame is given as a numpy array of floats in the range [0-1].

For testing new methods, the following images are provided:

Image | Description | Expected result
--- | --- | ---
00_a.png | a reference circle | reference value
01_a_space.png | the same reference circle in a larger image, with more space around it | approximately the reference value
02_2a.png | two of the reference circle | approximately twice the reference value
03_a_2x.png | the reference circle scaled 2x in each dimension | approximately the reference value
04_a_4x.png | the reference circle scaled 4x in each dimension | approximately the reference value
