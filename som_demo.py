import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pyERA.som import Som
from pyERA.utils import ExponentialDecay
from pyERA.utils import LinearDecay
import os

if __name__ == "__main__":
    SAVE_IMAGE = True
    output_path = "./output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    som_size = 512
    my_som = Som(matrix_size = som_size, input_size = 3, low = 0, high = 1, round_values = False)

    tot_epoch = 1000
    my_learning_rate = ExponentialDecay(starter_value = 0.4, decay_step = 50, decay_rate = 0.9, staircase = True)

    my_radius = ExponentialDecay(starter_value = np.rint(som_size/3), decay_step = 80, decay_rate = 0.90, staircase = True)

    for epoch in range(1, tot_epoch):
        if(SAVE_IMAGE == True):
            img = np.rint(my_som.return_weights_matrix()*255)
            plt.axis("off")
            plt.imshow(img)
            plt.savefig(output_path + str(epoch) + ".png", dpi = None, facecolor = 'black')

        learning_rate = my_learning_rate.return_decayed_value(global_step = epoch)
        radius = my_radius.return_decayed_value(global_step = epoch)
        colour_selected = np.random.randint(0, 6)
        colour_range = np.random.randint(100, 255)
        colour_range = float(colour_range)/255.0

        if(colour_selected == 0): input_vector = np.array([colour_range, 0, 0])
        if(colour_selected == 1): input_vector = np.array([colour_range, 0, 0])
        if(colour_selected == 2): input_vector = np.array([colour_range, 0, 0])
        if(colour_selected == 3): input_vector = np.array([colour_range, 0, 0])
        if(colour_selected == 4): input_vector = np.array([colour_range, 0, 0])
        if(colour_selected == 5): input_vector = np.array([colour_range, 0, 0])

        bmu_index = my_som.return_BMU_index(input_vector)
        bmu_weights = my_som.get_unit_weights(bmu_index[0], bmu_index[1])
        bmu_neighbourhood_list = my_som.return_unit_round_neighborhood(bmu_index[0], bmu_index[1], radius = radius)
        my_som.training_single_step(input_vector, units_list = bmu_neighbourhood_list, learning_rate = learning_rate, radius = radius, weighted_distance = False)
        print("Epoch: " + str(epoch))
        print("Learning Rate: " + str(learning_rate))
        print("Radius: " + str(radius))
        print("Input Vector: " + str(input_vector*255))
        print("BMU Index: " + str(bmu_index))
        print("BMU Weights: " + str(bmu_weights*255))

    img = np.rint(my_som.return_weights_matrix()*255)
    plt.axis("off")
    plt.imshow(img)
    plt.show()
    



