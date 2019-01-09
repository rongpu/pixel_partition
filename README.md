This repo includes Python scripts that divides the survey footprint into compact subregions with equal area, e.g. for jackknife subsampling. There are three steps:

1. Divide points (objects or randoms) in the survey footprint into healpix pixels;

2. Initial grouping using existing clustering algorithms (e.g. k-means clustering);

3. Randomly switch the labels of boundary pixels and check if the change improves the score (equal area and compact subregions). Iterate until the desired result is achived.

