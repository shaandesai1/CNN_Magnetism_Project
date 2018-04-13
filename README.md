# CNN_298R
CNN Magnetism Project for AC298R

This will outline all the necessary folders and files needed to run a convolutional neural network on charge densities.

Instructions to get setup:

Read through the evernote on GPU's in Odyssey (v2.0). The quickest way to setup is to initially just use CNN.py and folow the instructions to create a conda environment and install keras in it as well as load modules. the files you will need are CNN.py and a file called chgd_input which is linked here: https://www.dropbox.com/s/iquhjkpvhz7c7g6/chgdf_input?dl=0

Once you are done you can try using the same code in your login node and run sbatch cnn.batch which will run the same code in a non interactive environment. The great thing about sbatch is that you can edit your file run an sbatch and then repeat, it allows you to change variables in your local node and send jobs to worker nodes.

Once all of this works, use cnn_largep.py and cnn_largep.batch which will run a supercell and do everything on 2 GPU's. edit cnn_largep.py when you want to change filters etc.

See the benchmarking excel sheet of the different configurations already run. Seems like larger learning rates are better.

