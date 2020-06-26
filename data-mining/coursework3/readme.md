### Code Structure

***

I implemented DeepWalk and node2vec for link prediction. The code introduction is as follows:

* [main.ipynb](main.ipynb) is the main procedure.  
* [node2embedding.py](node2embedding.py) contains Implementation of DeepWalk and node2vec.
* [utils.py](utils.py) contains some useful functions.

I used the edges in the course3_edge.csv as positive edges, and I generated some negative edges. I used the node embedding vectors to get edge embedding vectors and then used these positive edges and negative edges to train a classifier. Thus, when getting the edge embedding vectors of test edges, we can use the trained classifier to classify the test edges into label {0, 1}.