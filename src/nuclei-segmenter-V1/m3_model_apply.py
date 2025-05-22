#######################
## m3_model_apply.py ##
#######################

## goal: module to apply model to new data

## author: Willem Bonnaffe (w.bonnaffe@gmail.com)

##############
## INITIATE ##
##############

## imports
import numpy as np
import os
import cv2
import torch

if __name__ == "__main__":

    ## absolute import 
    from f_UNet import UNet
    # from f_model import UNet
    from f_model import Net

else:

    ## relative import
    from .f_UNet import UNet
    # from .f_model import UNet
    from .f_model import Net

#
###

###############
## FUNCTIONS ##
###############

## tile
## goal: function to fold tensor into tiles
## X - numpy.array - array to tile
## n_tiles - int - number of tiles along the width and height of the array
def tile(X, n_tiles):
    return X.reshape(n_tiles*n_tiles, X.shape[1], int(X.shape[2]/n_tiles), int(X.shape[3]/n_tiles))

## untile
## goal: untile array
## X - numpy.array - array to tile
## n_tiles - int - number of tiles along the width and height of the array
def untile(X, n_tiles):
    return X.reshape(1, X.shape[1], X.shape[2]*n_tiles, X.shape[3]*n_tiles)

## normalise
## goal: normalise array between 0and 1
## array - np.array - to normalise
def normalise(array):
        return (array-np.min(array))/(np.max(array)-np.min(array))

## canvas embedding
## goal: embedd image in black square canvas
## img      - numpy array - image to embedd
## imgSize  - int         - size of image
## tileSize - int         - number of tiles to fit in the canvas
def embedd(img,imgSize,tileSize):
    nTiles     = (np.ceil(imgSize/tileSize)).astype("int")
    canvasSize = nTiles*tileSize
    if (canvasSize - img.shape[0] > 0):
        img = np.concatenate((img,np.zeros((canvasSize - img.shape[0],img.shape[1],img.shape[2]))),axis=0)
    if (canvasSize - img.shape[1] > 0):
        img = np.concatenate((img,np.zeros((img.shape[0],canvasSize - img.shape[1],img.shape[2]))),axis=1)
    return img

## model_predict
## goal: apply model to image to generate a segmentation profile
## img - np.array - image to segment
## model - torch  - model to apply
## device  - string - device to use for evaluating model
def model_predict(img, model, device):

    ## format image 
    X = img 
    X = X.reshape((1,X.shape[0],X.shape[1],X.shape[2]))
    X = X.swapaxes(3,2).swapaxes(2,1)
    X = torch.Tensor(X)
    X = X.to(device)
    
    ## get mask
    mask = (model(X)).cpu()
   
    ## format mask
    mask = torch.sigmoid(mask).detach().numpy()
    mask = mask.swapaxes(1,2).swapaxes(2,3)
    mask = mask[0] * 255

    ## terminate
    return mask

## model_list_load
## goal: load a list of models contained in a folder
## pt_models - string - path to folder that contains models
## device - string - which device to use
def model_list_load(pt_models, device):
    model_list = []
    rpt_models = os.listdir(pt_models)
    for rpt_model in rpt_models:
        model = UNet().to(device)
        model.load_state_dict(torch.load(pt_models + rpt_model, map_location=torch.device(device)))
        model.eval()
        model_list.append(model)
    return model_list

## model_list_predict
## goal: apply a list of models to image to generate an array of segmentation profiles
## img        - np.array - input image to the models
## model_list - list     - list of models to apply
## device     - string   - device to use for evaluating model
def model_list_predict(img, model_list, device):
    mask_list = []
    for model in model_list:
       mask_list.append(normalise(model_predict(img, model, device)))
    masks = np.stack(mask_list)
    return masks

## apply_model 
## goal: apply list of nuclei segmentation models to image tiles located in a folder
## pt_image_folder - string - path to input folder containing image tiles
## pt_output_folder - string - path to output folder to store output image tiles
## pt_model_folder - list of strings - paths to model to use to predict nuclei masks
## device - string - device to use for evaluating model
def apply_model(pt_image_folder, pt_output_folder, pt_model_folder, device = "cpu"):

    ## load models
    model_list = model_list_load(pt_model_folder, device) 

    ## check paths
    if os.path.exists(pt_output_folder) == False: 
        os.mkdir(pt_output_folder)

    ## for all input images 
    k = 0
    m = 0
    image_names = os.listdir(pt_image_folder) 
    for image_name in image_names: 
    
        ## iterator
        k = k + 1
        m = m + 1
        if (m/len(image_names) >= 0.1):
            print(str(int(k/len(image_names)*100))+"%")
            m = 0
    
        ## read image
        img = cv2.imread(pt_image_folder + image_name)
        
        ## skip predictions on background images
        if (np.std(img) > 5) & ((np.sum(img<10)/np.sum(img>0)) < 0.01):
    

            ## resize image if smaller than target_size
            if img.shape[0] < 1024:
                img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)

            ## apply models
            masks = model_list_predict(img, model_list, device)
    
            ## combine masks into one
            # mask = np.mean(masks, 0) # average
            mask = normalise(np.prod(masks, 0)) # prod
            # mask = normalise(np.sum(masks, 0))
    
            ## format mask
            mask = mask*255
    
        else:
            mask = np.zeros(img.shape)
       
        ## save mask
        cv2.imwrite(pt_output_folder + image_name, mask)

#
###

##########
## MAIN ##
##########

if __name__ == "__main__":

    ## imports
    import argparse
    
    ## arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_image_folder", default="/Volumes/SED/BDI/databases/segmentation/dev/debug_2/TB08.0970_V8_02/tiles/")
    parser.add_argument("--pt_output_folder", default="/Volumes/SED/BDI/databases/segmentation/dev/debug_2/TB08.0970_V8_02/predictions_nuclei/")
    parser.add_argument("--pt_model_folder", default="models_selected/")
    args = parser.parse_args()

    ## run main
    apply_model(pt_image_folder = args.pt_image_folder, pt_output_folder = args.pt_output_folder, pt_model_folder = args.pt_model_folder, device = "mps")


#
###
