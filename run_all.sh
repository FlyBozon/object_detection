# before running do this:
# chmod +x run_all.sh
# run it like:
# ./run_all.sh 50 <- where 50 is a number of random images for dataset u want to download

RANDOM_IMG_NUM="$1"

if [ -z "$RANDOM_IMG_NUM" ]; then
  echo "Usage: $0 <Random_img_number>"
  exit 1
fi

gnome-terminal --title="Create a datset" -- bash -c "
cd ~/object_detection;
echo 'Creating dataset';
python3 remove_background.py images_lab_opencv/srub for_dataset/srub;
echo 'srub background removed';
python3 remove_background.py images_lab_opencv/miecz for_dataset/miecz;
echo 'miecz background removed';
python3 remove_background.py images_lab_opencv/komb for_dataset/komb;
echo 'komb background removed';
echo 'downloading random images for dataset';
python3 download_random_images.py $RANDOM_IMG_NUM;
echo 'downloaded $RANDOM_IMG_NUM images for dataset';
python3 create_dataset.py ;
echo 'creates dataset (pasted objects on images)';
python3 split_into_datsets.py;
echo 'split images into training, validation and testing parts';
exec bash
"