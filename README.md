# GNN-mnist-workspace

The purpose of this project is to practice using GNNs to work with pixel-based data.

The first step of this project is to convert the MNIST dataset from pixel form to graph form. This step seems redundant as this dataset is readily available in graph form (in fact, it can be called in graph form within Spektral directly). However, I have chosen to do this step manually for two reasons: Firstly, it serves as practice for my research, which also involves converting pixels to graphs. Secondly, creating graphs manually makes it easier to change features of the graphs, and thus hopefully improve network performance.

At present, the DatasetGeneration class produces graphs similar to those that can be called directly within Spektral (https://graphneural.network/datasets/#mnist). The node features are the vectorized digits, except here the pixels are in a binary on/off state, where pixels with brightness > 0.4 in the original MNIST dataset are considered 'on' and all others are 'off' (this results in a loss of information, but was a conscious design choice to match my research. It can trivially be undone to match the Spektral dataset). Edges are created between adjacent 'on' nodes.

A visualization of the process is shown below:

<p align="center">
<img src="https://github.com/AWikramanayake/GNN-mnist-playground/blob/main/examples/MNIST%20base%20example.png?raw=true" />
</p>
<p align="center">
Figure 1: an example image from the MNIST dataset
</p>

<p align="center">
<img src="https://github.com/AWikramanayake/GNN-mnist-playground/blob/main/examples/MNIST%20binary%20example.png?raw=true" />
</p>
<p align="center">
Figure 2: the image after the pixels are set to the binary on/off state
</p>

<p align="center">
<img src="https://github.com/AWikramanayake/GNN-mnist-playground/blob/main/examples/MNIST%20graph%20example.png?raw=true)" />
</p>
<p align="center">
Figure 3: the resulting graph</p>
</p>

In addition to fine tuning the model and hyperparameters, the next steps consist of improving the features of the graph.
The current implementation notably lacks edge features.
One possible improvement would be to add edges to more distant neighbours, with the edges weighted by distance.

<p align="center">
<img src="https://github.com/AWikramanayake/GNN-mnist-playground/blob/main/examples/MNIST%20graph%20extended%20example.png?raw=true" alt="Sublime's custom image"/>
</p>
<p align="center">
Figure 4:  a graph with additional longer edges
</p>

However, introducing this many edges may be computationally prohibitive, so finding other potential edge and node features might be a more prudent approach. And of course, as other work in this area has shown, creating nodes that do not correspond to pixels 1:1 may be an even better approach [1].


[1]: Monti, F., Boscaini, D., Masci, J., Rodola, E., Svoboda, J., &amp; Bronstein, M. M. (2017). Geometric deep learning on graphs and manifolds using mixture model cnns. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/cvpr.2017.576 
