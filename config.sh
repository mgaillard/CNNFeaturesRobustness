# A path to the directory containing the images
image_dir="raw_images"

# A path in which the transformed images will be stored
tmp_image_dir="images"

# Example of directory structure of $tmp_image_dir
# ├── $tmp_image_dir
# |   ├── base
# |   ├── blur
# |   ├── gray
# |   ├── resize50
# |   ├── compress10
# |   ├── rotate5
# |   └── crop10

declare -a transformations=("base" "blur" "gray" "resize50" "compress10" "rotate5" "crop10")
# Possible models:
# InceptionV3_predictions
# InceptionV3_mixed10_avg
# InceptionV3_mixed10_max
# InceptionV3_mixed9_avg
# InceptionV3_mixed9_max
# InceptionV3_mixed9_max_pca_64
# InceptionV3_mixed8_avg
# InceptionV3_mixed8_max
# InceptionV3_mixed7_avg
# InceptionV3_mixed7_max
# InceptionV3_mixed6_avg
# InceptionV3_mixed6_max
# InceptionV3_mixed5_avg
# InceptionV3_mixed5_max
# InceptionV3_mixed4_avg
# InceptionV3_mixed4_max
# InceptionV3_mixed3_avg
# InceptionV3_mixed3_max
# InceptionV3_mixed2_avg
# InceptionV3_mixed2_max
# InceptionV3_mixed1_avg
# InceptionV3_mixed1_max
# InceptionV3_mixed0_avg
# InceptionV3_mixed0_max
# Xception_predictions
# Xception_block14_sepconv2_act_avg
# Xception_block14_sepconv2_act_max
# Xception_block14_sepconv1_act_avg
# Xception_block14_sepconv1_act_max
# Xception_add_12_avg
# Xception_add_12_max
# Xception_add_11_avg
# Xception_add_11_max
# Xception_add_10_avg
# Xception_add_10_max
# Xception_add_9_avg
# Xception_add_9_max
# Xception_add_8_avg
# Xception_add_8_max
# Xception_add_7_avg
# Xception_add_7_max
# Xception_add_6_avg
# Xception_add_6_max
# Xception_add_5_avg
# Xception_add_5_max
# Xception_add_4_avg
# Xception_add_4_max
# Xception_add_3_avg
# Xception_add_3_max
# Xception_add_2_avg
# Xception_add_2_max
# Xception_add_1_avg
# Xception_add_1_max
# ResNet50_predictions
# ResNet50_flatten_1
# ResNet50_flatten_1_norm_l2
# ResNet50_avg_pool_avg
# ResNet50_avg_pool_avg_norm_l2
# ResNet50_avg_pool_max
# ResNet50_avg_pool_max_norm_l2
# ResNet50_activation_46_avg
# ResNet50_activation_46_max
# ResNet50_activation_46_max_pca_64
# ResNet50_activation_43_avg
# ResNet50_activation_43_max
# ResNet50_activation_43_max_pca_64
# ResNet50_activation_40_avg
# ResNet50_activation_40_max
# ResNet50_activation_37_avg
# ResNet50_activation_37_max
# ResNet50_activation_34_avg
# ResNet50_activation_34_max
# ResNet50_activation_31_avg
# ResNet50_activation_31_max
# ResNet50_activation_28_avg
# ResNet50_activation_28_max
# ResNet50_activation_22_avg
# ResNet50_activation_22_max
# ResNet50_activation_19_avg
# ResNet50_activation_19_max
# ResNet50_activation_16_avg
# ResNet50_activation_16_max
# ResNet50_activation_13_avg
# ResNet50_activation_13_max
# ResNet50_activation_10_avg
# ResNet50_activation_10_max
# ResNet50_activation_7_avg
# ResNet50_activation_7_max
# ResNet50_activation_4_avg
# ResNet50_activation_4_max
# VGG16_predictions
# VGG16_fc2
# VGG16_fc2_norm_l2
# VGG16_fc1
# VGG16_fc1_norm_l2
# VGG16_flatten
# VGG16_flatten_norm_l2
# VGG16_block5_pool_avg
# VGG16_block5_pool_avg_norm_l2
# VGG16_block5_pool_max
# VGG16_block5_pool_max_norm_l2
# VGG16_block5_pool_max_pca_64
# VGG16_block5_pool_max_pca_64_norm_l2
# VGG16_block4_pool_avg
# VGG16_block4_pool_avg_norm_l2
# VGG16_block4_pool_max
# VGG16_block4_pool_max_norm_l2
# VGG16_block3_pool_avg
# VGG16_block3_pool_avg_norm_l2
# VGG16_block3_pool_max
# VGG16_block3_pool_max_norm_l2
# VGG19_predictions
# VGG19_fc2
# VGG19_fc2_norm_l2
# VGG19_fc1
# VGG19_fc1_norm_l2
# VGG19_flatten
# VGG19_flatten_norm_l2
# VGG19_block5_pool_avg
# VGG19_block5_pool_avg_norm_l2
# VGG19_block5_pool_max
# VGG19_block5_pool_max_norm_l2
# VGG19_block5_pool_max_pca_64
# VGG19_block5_pool_max_pca_64_norm_l2
# VGG19_block4_pool_avg
# VGG19_block4_pool_avg_norm_l2
# VGG19_block4_pool_max
# VGG19_block4_pool_max_norm_l2
# VGG19_block3_pool_avg
# VGG19_block3_pool_avg_norm_l2
# VGG19_block3_pool_max
# VGG19_block3_pool_max_norm_l2
declare -a cnn_model=("VGG16_block5_pool_avg" "VGG16_block5_pool_max")
# Possible distances: 'euclidean', 'euclidean_square', 'cosine'
declare -a cnn_features_distances=("euclidean" "cosine")

# Set to "true" to perform a benchmark on individual transformations
individual_transformation_benchmark=true

# Set to "true" to performa a benchmark on all modifications at the same time
multiple_transformation_benchmark=true

declare -A transformation_name
transformation_name["base"]="Base"
transformation_name["blur"]="Blur"
transformation_name["gray"]="Grayscale"
transformation_name["resize5"]="Resize 5%"
transformation_name["resize10"]="Resize 10%"
transformation_name["resize25"]="Resize 25%"
transformation_name["resize50"]="Resize 50%"
transformation_name["resize75"]="Resize 75%"
transformation_name["compress1"]="Compress with quality 1"
transformation_name["compress5"]="Compress with quality 5"
transformation_name["compress10"]="Compress with quality 10"
transformation_name["compress20"]="Compress with quality 20"
transformation_name["compress40"]="Compress with quality 40"
transformation_name["compress80"]="Compress with quality 80"
transformation_name["rotate1"]="Rotation with angle 1"
transformation_name["rotate2"]="Rotation with angle 2"
transformation_name["rotate3"]="Rotation with angle 3"
transformation_name["rotate4"]="Rotation with angle 4"
transformation_name["rotate5"]="Rotation with angle 5"
transformation_name["crop4"]="Crop right 4%"
transformation_name["crop8"]="Crop right 8%"
transformation_name["crop10"]="Crop right 10%"
transformation_name["crop12"]="Crop right 12%"
transformation_name["crop16"]="Crop right 16%"
transformation_name["crop20"]="Crop right 20%"

declare -A transformation_command
transformation_command["base"]=""
transformation_command["blur"]="mogrify -path $tmp_image_dir/blur -blur 4x2 $tmp_image_dir/base/*"
transformation_command["gray"]="mogrify -path $tmp_image_dir/gray -type Grayscale $tmp_image_dir/base/*"
transformation_command["resize5"]="mogrify -path $tmp_image_dir/resize5 -resize 5% $tmp_image_dir/base/*"
transformation_command["resize10"]="mogrify -path $tmp_image_dir/resize10 -resize 10% $tmp_image_dir/base/*"
transformation_command["resize25"]="mogrify -path $tmp_image_dir/resize25 -resize 25% $tmp_image_dir/base/*"
transformation_command["resize50"]="mogrify -path $tmp_image_dir/resize50 -resize 50% $tmp_image_dir/base/*"
transformation_command["resize75"]="mogrify -path $tmp_image_dir/resize75 -resize 75% $tmp_image_dir/base/*"
transformation_command["compress1"]="mogrify -path $tmp_image_dir/compress1 -format jpg -quality 1 $tmp_image_dir/base/*"
transformation_command["compress5"]="mogrify -path $tmp_image_dir/compress5 -format jpg -quality 5 $tmp_image_dir/base/*"
transformation_command["compress10"]="mogrify -path $tmp_image_dir/compress10 -format jpg -quality 10 $tmp_image_dir/base/*"
transformation_command["compress20"]="mogrify -path $tmp_image_dir/compress20 -format jpg -quality 20 $tmp_image_dir/base/*"
transformation_command["compress40"]="mogrify -path $tmp_image_dir/compress40 -format jpg -quality 40 $tmp_image_dir/base/*"
transformation_command["compress80"]="mogrify -path $tmp_image_dir/compress80 -format jpg -quality 80 $tmp_image_dir/base/*"
transformation_command["rotate1"]="mogrify -path $tmp_image_dir/rotate1 -rotate 1 $tmp_image_dir/base/*"
transformation_command["rotate2"]="mogrify -path $tmp_image_dir/rotate2 -rotate 2 $tmp_image_dir/base/*"
transformation_command["rotate3"]="mogrify -path $tmp_image_dir/rotate3 -rotate 3 $tmp_image_dir/base/*"
transformation_command["rotate4"]="mogrify -path $tmp_image_dir/rotate4 -rotate 4 $tmp_image_dir/base/*"
transformation_command["rotate5"]="mogrify -path $tmp_image_dir/rotate5 -rotate 5 $tmp_image_dir/base/*"
transformation_command["crop4"]="mogrify -path $tmp_image_dir/crop4 -crop 96%x100%+0+0 $tmp_image_dir/base/*"
transformation_command["crop8"]="mogrify -path $tmp_image_dir/crop8 -crop 92%x100%+0+0 $tmp_image_dir/base/*"
transformation_command["crop10"]="mogrify -path $tmp_image_dir/crop10 -crop 90%x100%+0+0 $tmp_image_dir/base/*"
transformation_command["crop12"]="mogrify -path $tmp_image_dir/crop12 -crop 88%x100%+0+0 $tmp_image_dir/base/*"
transformation_command["crop16"]="mogrify -path $tmp_image_dir/crop16 -crop 84%x100%+0+0 $tmp_image_dir/base/*"
transformation_command["crop20"]="mogrify -path $tmp_image_dir/crop20 -crop 80%x100%+0+0 $tmp_image_dir/base/*"

declare -A cnn_model_threshold_start
declare -A cnn_model_threshold_end
declare -A cnn_model_threshold_step


function trace {
    echo $1 ; date +"%m/%d/%Y %H:%M:%S" ; echo ""
}
