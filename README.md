# RL_PATH_TRACING
Find the source code in directory "time".

"time/training" contains utilitary files.

Generatin of the dataset:

Move the datasets from the following link: https://drive.google.com/drive/folders/1tTb8vQcfsH7FbHgRnOOW5WUzNgAoFCjc?usp=sharing to the "datasets" repository.

Install blender (we used version 3.2.2 and 3.5.1).

Configure blender such that it can generate additional data and motion vectors (see datasets/tuto.pdf)

For every dataset:

-Run script1.py, which will generate the additonal data, generate the ground truth and individual sample images, generate the motion vectors and convert them in the right format.

