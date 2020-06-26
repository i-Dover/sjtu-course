### Code Structure

***

I implemented Louvain and DeepWalk Algorithm for this community detection task. The code instructions are as follows:

* [louvain.py](louvain.py) is the implementation of the Louvain algorithm.
* [deepwalk.py](deepwalk.py) is the implement of the DeepWalk algorithm. 
* [utils.py](utils.py) contains some useful functions.
* [deep_walk.ipynb](deep_walk.ipynb) is the runnable script for running the DeepWalk algorithm.
* [louvain_main.ipynb](louvain_main.ipynb) is the runnable script for running the Louvain algorithm.

The final test accuracy by Louvain is from 0.99496 to 0.99839, and the final test accuracy by DeepWalk is from 0.99850 to 0.99946. The best result I got was 0.99946. 

Just try to run [deep_walk.ipynb](deep_walk.ipynb) and [louvain_main.ipynb](louvain_main.ipynb), the final accuracy may fluctuate a little, which may be different from the best accuracy I got in the LeaderBoard.