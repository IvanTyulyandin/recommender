# Recommender system

This is a simple recommender system, based on cosine similarity of users.
Based on data from https://labrosa.ee.columbia.edu/millionsong/tasteprofile (triplets for 1M users(~500MB)).


## How to build
Program uses GNU parallel extension, so you need GCC to build system, tested with 6.2.0 and 7.3.0 GCC versions.


**chmod +x build.sh \
./build.sh**


## How to use
To run metrics (RMSE, nDCG and Gini): \
\
**./recommender [/path/to/triplets]** \
\
Default path for triplets is the same directory where system was built.

If you want to predict concrete songs for user,
you need to call *predictFromNeighborsWithSongMark* or *predictSongsFromNearestNeighbors*
from recommender.h