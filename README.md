# FastGeodis: Fast Generalised Geodesic Distance Transform

We added pairwise Geodesic Distance matrix computation to the original FastGeodis codebase(https://github.com/masadcv/FastGeodis). 

# Installation

Move to the folder and print "python setup.py install" to build and install this package.

# Usage

The pairwise_geodesic2d takes and image with size (1, c, h, w) and several other parameters as input. 
Notice the input image should be of batch size 1. If you have an image with batch size more than 1, you should iterate it along batch size and call this function.

```python
import FastGeodis
matrix = FastGeodis.pairwise_geodesic2d(image, v, lamb, iterations)
```
