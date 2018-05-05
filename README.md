# CNN_299R
<b>CNN Magnetism Project for AC299R - Shaan Desai</b>
<b>Using convolutional neural networks to understand magnetism in two-dimensional ferromagnetic structures based on Cr<sub>2</sub>Ge<sub>2</sub>Te<sub>6</sub> - Shaan Desai</b>

## Introduction

Over the past decade, two-dimensional materials have demonstrated immense potential for creating advances in fundamental research and industrial applications. In particular, experimental and computational studies of the transition-metal based compounds [1,2] both highlight the emergence of numerous magnetic phases in two-dimensional crystals. These findings have sparked significant excitement in materials science for their potential use in data storage and spintronics applications. A notable example of this is Cr<sub>2</sub>Ge<sub>2</sub>Te<sub>6</sub> which was recently experimentally shown to exhibit ferromagnetism [3]. Despite these experimental observations and our general understanding of how magnetism arises in bulk crystals, we have yet to fully understand the specific mechanisms through which magnetism arises in these two-dimensional (2-D) materials. Developing this understanding could significantly impact (1), our ability to predict the properties of novel materials - new materials which are designed by decorating atomic sites of known 2-D structures with different atoms [fig 1] and (2), our ability to engineer materials for specific device applications. 






The code in this repo allows you to run a Convolutional Neural Network with 3D charge density profiles as input. The code is configured to run both locally and on Harvard FAS RC GPU's.

<b>Workflow</b>

1. Pull all composite DFT calculations into one folder using chgcar.py when collecting data from the supercomputer that runs DFT calcs.
2. Run magdensity splitter on CHGCAR files to split (spin up + spin down) from (spin up - spin down) and store the latter in a magnetic densities folder.
3. Run the scraper notebook to save the magnetic densities and response variable (magnetic moment) into a large dataframe.
4. To run the convolution locally use the notebook, it basically configures a simple 2/3D convolution of the inputs and is a great way to test whether you have the right files. Make sure the convolutions have small filter sizes (e.g. 5,5,5) since larger weights require a lot of memory and CPU's aren't good for this. 
5. You can play with the architecture and will soon realize the code runs very slowly (especially if using Conv3D), as such we need to run this on GPU's:

<i>Instructions to get setup using a GPU</i>

Read through the evernote on GPU's in Odyssey (v2.0). The quickest way to setup is to initially just use CNN.py and folow the instructions to create a conda environment and install keras in it as well as load modules. Initially the file you will need is tester.py which is a CNN that runs on the MNIST dataset. You will have to use srun to run the code or you can create a small sbatch file to run it.

Only once the above works can you move forward!
You should now be ready to run our CNN. Configure the CNN in CNN.py to your liking and download a file called chgd_input which is linked here: https://www.dropbox.com/s/iquhjkpvhz7c7g6/chgdf_input?dl=0 into the same directory into your local node on Odyssey. Also download cnn.batch and configure the filenames you want to store. Once done, simply run <sbatch cnn.batch>. The great thing about sbatch is that you can edit your file, run an sbatch, and then repeat. It allows you to change variables in your local node and send jobs with the new parameters to worker nodes.

Once all of this works, use cnn_largep.py and cnn_largep.batch which will run a supercell and do everything on 2 GPU's. edit cnn_largep.py when you want to change filters etc.

See the benchmarking excel sheet of the different configurations already run and results I can provide. Seems like larger learning rates are better.

6. After running your code, you can copy the h5 file output into your local machine and run visualizer.ipynb which will let you see projections of your filter.


<b>Notes on Architecture</b>

- The point of running this model is to be able to capture patterns such as Nearest Neighbor exchange in filters. Note that Sivadas et al show that these occur on the order of atomic lengths which means we should probably scale our unit cell to a supercell (done in cnn_largep.py in function matmul) as well as increase our filter sizes to greater than 60 in each dimension.
- We need to think carefully about the classification model we are building. At present, we are classifying composites as 1's if they have a bohr magneton > 4. However, this cutoff was determined using a histogram.
- At present we believe we should have 3 filters because this is consistent with Sivadas et al and the number of patterns they present, however, it might be the case that more are needed to capture superexchange.
- Need to figure out how to get granularity.
- Need to understand why a large learning rate yields a more interpretable pattern.
- Need more datab

