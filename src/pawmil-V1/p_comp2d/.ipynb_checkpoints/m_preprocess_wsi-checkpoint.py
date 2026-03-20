#####
## ##
#####

##############
## INITIATE ##
##############

## Imports
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import argparse

## Import modules
from .f_utils import format_path_file
from .f_utils import format_label_file
from .f_wsi_reader import get_tile_map
from .f_wsi_reader import slide_to_tiles_at_coordinates
from .f_wsi_reader import read_slide
from .f_wsi_reader import get_dimensions
from .f_wsi_reader import apply_all_filters
from .f_wsi_reader import apply_all_transforms
from .f_wsi_reader import reassemble_tiles

## WIP >>
# try: # WIP: find better way of switching to isyntax reader
#     from f_wsi_reader_isyntax import get_tile_map
#     from f_wsi_reader_isyntax import slide_to_tiles_at_coordinates
#     from f_wsi_reader_isyntax import read_slide
#     from f_wsi_reader_isyntax import get_dimensions
# except:
## WIP <<

#
###

###############
## FUNCTIONS ##
###############

def main(pt_input_folder, paths_file, labels_file, tile_size, level):
    """
    """
    #############################
    ## USER DEFINED PARAMETERS ##

    ## Paths and input files
    input_folder = pt_input_folder # e.g. "/Volumes/SED/BDI/projects/PAW-MIL_2023/b0_0_3/PPCG-BCR-HQ/"
    output_folder = input_folder # args.pt_output_folder 
    paths_list = format_path_file(input_folder + paths_file)
    labels = format_label_file(input_folder + labels_file)
    
    ## Parameters
    tile_size = tile_size
    level = level
    extract_tiles = False

    ############## 
    ## INITIATE ##

    ## Create output folders
    if os.path.exists(output_folder) == False:
        os.mkdir(output_folder)
    output_subfolder_2 = output_folder + "preprocessed/"
    if os.path.exists(output_subfolder_2) == False:
        os.mkdir(output_subfolder_2)
    output_subfolder = output_subfolder_2 + "thumbnails/"
    if os.path.exists(output_subfolder) == False:
        os.mkdir(output_subfolder)
 
    ## Containers
    slide_size_list = []
    tile_map_list = []
    tile_map_masked_list = []
    tile_coordinates_list = []
    
    ## For all files
    l = 1
    for svs_path in paths_list:
    
        ## Iterator
        print(f"sample {l}/{len(paths_list)}")
        print(svs_path)
        
        ################## 
        ## GET TILE MAP ##
    
        ## Open slide
        slide = read_slide(svs_path)
        slide_size = get_dimensions(slide)
    
        ## Get tile_map 
        t0 = time.time()
        thumbnail, tile_map = get_tile_map(slide, tile_size)
        tf = time.time()
        print(f"tile map created in {tf - t0:.2f}s")
        
        ## Plot RGB of tile_map
        fig, axs = plt.subplots(3,3)
        for i in range(3):
            for j in range(3):
                if i != j:
                    x = tile_map[:,:,i]
                    y = tile_map[:,:,j]
                    axs[j,i].scatter(x, y)
        plt.savefig(output_subfolder + f"{l}_fig_tile_map_rgb_{l}.png")
        plt.close()
    
        ## Visualise thumbnail
        plt.imshow(thumbnail)
        plt.savefig(output_subfolder + f"{l}_fig_thumbnail_{l}.png")
        plt.close()
    
        ## Visualise tile_map
        plt.imshow(tile_map)
        plt.savefig(output_subfolder + f"{l}_fig_tile_map_{l}.png")
        plt.close()
    
        ##########################
        ## SEGMENT TISSUE TILES ##
    
        ## Extract coordinates of pixels in tile_map that contain tissue
        mask = apply_all_filters(tile_map)
        mask = apply_all_transforms(mask)
       
        ## Visualise mask
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        tile_map_masked = tile_map * mask
        plt.imshow(tile_map_masked)
        plt.savefig(output_subfolder + f"{l}_fig_tile_map_masked_{l}.png")
        plt.close()
        mask = mask[:,:,0]
    
        ## Get coordinates of tiles with tissue
        tile_coordinates = (np.argwhere(mask.T > 0) * tile_size).astype(int)
        
        ####################
        ## RETRIEVE TILES ##
        if extract_tiles == True:
        
            ## Extract tiles from coordinates
            t0 = time.time()
            tiles = slide_to_tiles_at_coordinates(slide, tile_coordinates, tile_size, level)
            tf = time.time()
            print(f"extracted tiles in {tf - t0:.2f}s")
            
            ## Visualise tiles
            k = 0
            num_plots = 4
            fig, axs = plt.subplots(num_plots, num_plots)
            for i in range(num_plots):
                for j in range(num_plots):
                    axs[j,i].imshow(tiles[k])
                    k += 1
            plt.savefig(output_subfolder + f"{l}_fig_tiles_{l}.png")
            plt.close()
            
            ## Check image by reassembling tiles
            wsi_img_reassembled = reassemble_tiles(tiles, tile_coordinates)
            wsi_img_reassembled = cv2.resize(wsi_img_reassembled, (int(wsi_img_reassembled.shape[1]/10), int(wsi_img_reassembled.shape[0]/10)))
    
            ## Visualise reassembled slide
            plt.imshow(wsi_img_reassembled)
            plt.savefig(output_subfolder + f"{l}_fig_slide_reassembled_{l}.png")
            plt.close()
    
        ###############
        ## TERMINATE ##
    
        ## Iterate
        l = l + 1
    
        ## Store results
        slide_size_list.append(slide_size)
        tile_map_list.append(tile_map)
        tile_map_masked_list.append(tile_map_masked)
        tile_coordinates_list.append(tile_coordinates)

    ## Save results
    with open(output_subfolder_2 + 'obj_coordinates.pkl', 'wb') as file:
        pickle.dump(tile_coordinates_list, file)

    ####################
    ## VISUALISATIONS ## 

    ## Summary statistics 
    q_tile_size_x_list = []
    q_tile_size_y_list = []
    for tile_map in tile_map_list:
        q_tile_size_x_list.append(tile_map.shape[0])
        q_tile_size_y_list.append(tile_map.shape[1])
    q_tile_number_list = []
    for tile_map_masked in tile_map_masked_list:
        q_tile_number_list.append(((tile_map_masked>0)*1).sum())
    np.savetxt(output_subfolder_2 + "txt_tile_map_sizes_x.txt", np.array(q_tile_size_x_list))
    np.savetxt(output_subfolder_2 + "txt_tile_map_sizes_y.txt", np.array(q_tile_size_y_list))
    np.savetxt(output_subfolder_2 + "txt_tile_map_numbers.txt", np.array(q_tile_number_list))
    
    ## Visualise slide size
    fig, axs = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            if i != j:
                for tile_map in tile_map_list:
                    x = tile_map[:,:,i]
                    y = tile_map[:,:,j]
                    axs[j,i].scatter(x, y)
    plt.savefig(output_subfolder_2 + f"fig_tile_map_rgb_all.png")
    plt.close()
     
    ## Visualise slide size
    for tile_map in tile_map_list:
        plt.scatter(tile_map.shape[0], tile_map.shape[1])
    plt.savefig(output_subfolder_2 + f"fig_tile_map_size_all.png")
    plt.close()
    
    ## Visualise slide size
    for tile_map_masked in tile_map_masked_list:
        plt.scatter(((tile_map_masked>0)*1).sum(), ((tile_map_masked>0)*1).sum())
    plt.savefig(output_subfolder_2 + f"fig_tile_map_masked_size_all.png")
    plt.close()
    
    ## Visualise tile map size vs number of tiles
    for i in range(len(tile_map_list)):
        x = tile_map_list[i].shape[0] * tile_map_list[i].shape[1]
        y = np.sum(tile_map_masked_list[i] > 0)
        plt.scatter(x, y)
    plt.savefig(output_subfolder_2 + f"fig_slide_size_vs_num_tiles.png")
    plt.close()

#
###
