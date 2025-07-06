set -e

docker build -f dockerfile \
    --build-arg work_folder=$PWD/.. \
    -t instructflip .
