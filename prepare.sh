#!/bin/bash

source config.sh

# Result directory
mkdir results

# Compile the program to compute the distance distributions
mkdir features_distances_release
cd features_distances_release
cmake -DCMAKE_BUILD_TYPE=Release ../features_distances
make
cd ..

# Transform images
mkdir "$tmp_image_dir"
mkdir "$tmp_image_dir/base"
cp -a "$image_dir/." "$tmp_image_dir/base/"

mkdir "$tmp_image_dir/blur"
mogrify -path "$tmp_image_dir/blur" -blur 4x2 "$tmp_image_dir/base/*"

mkdir "$tmp_image_dir/gray"
mogrify -path "$tmp_image_dir/gray" -type Grayscale "$tmp_image_dir/base/*"

mkdir "$tmp_image_dir/resize50"
mogrify -path "$tmp_image_dir/resize50" -resize 50% "$tmp_image_dir/base/*"

mkdir "$tmp_image_dir/compress10"
mogrify -path "$tmp_image_dir/compress10" -format jpg -quality 10 "$tmp_image_dir/base/*"

mkdir "$tmp_image_dir/rotate5"
mogrify -path "$tmp_image_dir/rotate5" -rotate 5 "$tmp_image_dir/base/*"

mkdir "$tmp_image_dir/crop10"
mogrify -path "$tmp_image_dir/crop10" -crop 90%x100%+0+0 "$tmp_image_dir/base/*"

# Extract the features
mkdir features
for t in "base" "blur" "gray" "resize50" "compress10" "rotate5" "crop10"; do
    echo "$tmp_image_dir/$t"
    python3 features_extractor/main.py --output "features/features_$t.h5" --format h5 "$tmp_image_dir/$t"
done
