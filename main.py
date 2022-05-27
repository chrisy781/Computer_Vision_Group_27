from preprocess import filter_imgs_on_label_size, load_data, plot_25_images, format_labels, save_data, blur_images

# get the path/directory
folder_dir = "/home/stijn/Robotics/Computer Vision w DL/Yolo/obj"

images, labels = load_data(folder_dir)
occupied_area_by_ball = 0.5 # normalized value, so 0.5 means that 50 percent of the image has to be occupied by the BOUNDING BOX around the golfball

large_images, large_labels, small_images, small_labels = filter_imgs_on_label_size(images, labels, occupied_area_by_ball)
labels = format_labels(large_labels)

#save_data(images, labels)
large_images = blur_images(large_images)
#plot_25_images(large_images)