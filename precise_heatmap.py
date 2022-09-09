import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Return a list of boxes for all the images in [x0,y0,x1,y1] format
def get_all_boxes(nb_img):
    
    coord_list = []
    
    for index in range(nb_img):
        df = pd.read_json('Marseille ('+ str(index+1) +').json', orient='index')
        
        object_list = df.values[3][0]
        
        for item in object_list:
            if item['classTitle'] == 'People':
                coord_list.append(item['points']['exterior'][0]+item['points']['exterior'][1])

    return coord_list


#Return a color matrix for the heatmap
def get_color_matrix(coord_list):
    df = pd.read_json('Marseille (' + str(10) +').json', orient='index')
    
    img_size = df.values[2][0]
    img_width,img_height = img_size['width'],img_size['height']
    
    color_boxes=[0]*img_width*img_height

    for rect_coord in coord_list:
        for x in range(rect_coord[0],rect_coord[2]+1):
            for y in range(rect_coord[1],rect_coord[3]+1):
                if ((y-1)*img_width+x-1<img_width*img_height):
                    color_boxes[(y-1)*img_width+x-1]+=1
                    
    return color_boxes, img_width, img_height

def plot_heatmap(nb_img):
    coord_list = get_all_boxes(nb_img)
    color_boxes,img_width, img_height = get_color_matrix(coord_list)



    heatmap = np.array(color_boxes).reshape((img_height,img_width))


    img = cv2.imread("Marseille_img.jpg")  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    fig = plt.figure(figsize=(50,50))

    ax2 = plt.subplot(1, 4, 3, )
    ax2.imshow(np.squeeze(img), alpha = 0.5)
    plt.axis('off')
    hm = ax2.matshow(heatmap, alpha = 0.5, cmap='seismic', interpolation='nearest')

#plot_heatmap(776) for Marseille
#plot_heatmap(142) for Roissy
#plot_heatmap(97) for Biosav
#plot_heatmap(212) for Devisubox_sec