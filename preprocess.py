from cv2 import COLOR_HSV2BGR_FULL
import numpy as np
import matplotlib.pyplot as plt
import pathlib 
import cv2
import os


def load_data(path):
    images = []
    labels = []
    for data in os.listdir(path):
    
        # check if the image ends with png
        if (data.endswith(".jpg")):
            images.append(data)
            labels.append(data.replace(".jpg",".txt"))
    return images, labels

def get_random_data_point(data):
    return np.random.randint(0, len(data))

def process_data_point(img, label):
    path_txt = pathlib.Path("obj/{}".format(label))

    with open(path_txt, 'r') as f:
        lines = f.readlines()

    locations = []

    for line in lines:
        line = line.replace('\n', '').split()
        coordinates = []
        for coordinate in line:
            coordinates.append(float(coordinate))
        coordinates = np.array(coordinates)
        locations.append(coordinates)

    img = cv2.imread("obj/{}".format(img))
    for location in locations:
        _, x, y, w, h = location.ravel()
        x = img.shape[0]*x
        y = img.shape[0]*y
        w = img.shape[1]*w
        h = img.shape[1]*h
        img = cv2.rectangle(img, (int(x - 0.5*w), int(y - 0.5*h)), (int(x + 0.5*w), int(y + 0.5*h)), (0,0,255), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def process_label(label):
    path_txt = pathlib.Path("obj/{}".format(label))

    with open(path_txt, 'r') as f:
        lines = f.readlines()

    locations = []
    for line in lines:
        line = line.replace('\n', '').split()
        coordinates = []
        for coordinate in line:
            coordinates.append(float(coordinate))
        coordinates = np.array(coordinates)
        locations.append(coordinates)
        return locations

def format_labels(labels):
    formatted_labels = []
    for label in labels:
        formatted_labels.append(format_label(label))
    return formatted_labels

def format_label(label):
    total = ""
    for location in label:
        str_loc = ""
        for i, coordinate in enumerate(location):
            if i == 0:
                coordinate = int(coordinate)
            val = str(coordinate)+ " "
            str_loc = str_loc+val
        str_loc = str_loc+"\n"
        total = total + str_loc
    total = total.replace("[","").replace("]","")
    return total    

def save_data(images, labels):
    for i, label in enumerate(labels):
        path_txt = pathlib.Path("filtered_obj/{}.txt".format(i))
        path_img = pathlib.Path("filtered_obj/{}.jpg".format(i))

        with open(path_txt, 'x') as f:
            f.write(label)
            f.close()
        
        image = plt.imread("obj/{}".format(images[i]))

        plt.imsave(path_img, image)

def process_labels(labels):
    locations = []
    for label in labels:
        locations.append(process_label(label))
    return locations

def plot_random_data_points_with_label(images, labels):
    fig = plt.figure(figsize=(8, 8))
    plt.suptitle("Subset of the dateset")
    plt.axis('off')
    columns = 5
    rows = 5
    ax = []

    for i in range(rows*columns):
        a = get_random_data_point(images)
        plot = process_data_point(images[a], labels[a])
        ax.append(fig.add_subplot(rows, columns, i+1))
        plt.axis('off')
        ax[-1].set_title(str(i+1))  # set title
        plt.imshow(plot)
    plt.show()

def plot_25_images(images):
    fig = plt.figure(figsize=(8, 8))
    plt.suptitle("Subset of the dateset")
    plt.axis('off')
    columns = 5
    rows = 5
    ax = []
    for i in range(rows*columns):   
        image = plt.imread("obj/{}".format(images[i]))
        ax.append(fig.add_subplot(rows, columns, i+1))
        plt.axis('off')
        ax[-1].set_title(str(i+1))  # set title
        plt.imshow(image)
    plt.show()

def filter_imgs_on_label_size(images, labels, surface_area):
    large_labels = []
    large_images = []
    small_labels = []
    small_images = []
    orig_size = len(images)
    labels = process_labels(labels)
    for i, label in enumerate(labels):
        if label:
            for location in label:
                _, x, y, w, h = location.ravel()
                if w*h >= surface_area:
                    large_labels.append(label)
                    large_images.append(images[i])
                elif w*h < surface_area:
                    small_labels.append(label)
                    small_images.append(images[i])
    new_size = len(large_images)
    percentage = (1-new_size/orig_size)*100
    print("Small set is {percentage:.1f}".format(percentage=percentage), " %"," of the origanal set")
    return large_images, large_labels, small_images, small_labels

def blur_images(images):
    ksize = (10, 10)
    blurred_imgs = []
    for image in images:
        print(image)
        imaged = plt.imread("obj/{}".format(image))
        blurred_imgs.append(cv2.blur(imaged, ksize))
    return blurred_imgs