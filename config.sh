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
declare -a cnn_model=("VGG16_block5_pool_avg" "VGG16_block5_pool_max")
cnn_features_distance="euclidean_square"

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
cnn_model_threshold_start["VGG16_block5_pool_avg"]=100
cnn_model_threshold_start["VGG16_block5_pool_max"]=5000

declare -A cnn_model_threshold_end
cnn_model_threshold_end["VGG16_block5_pool_avg"]=10000
cnn_model_threshold_end["VGG16_block5_pool_max"]=500000

declare -A cnn_model_threshold_step
cnn_model_threshold_step["VGG16_block5_pool_avg"]=100
cnn_model_threshold_step["VGG16_block5_pool_max"]=5000

function trace {
    echo $1 ; date +"%m/%d/%Y %H:%M:%S" ; echo ""
}