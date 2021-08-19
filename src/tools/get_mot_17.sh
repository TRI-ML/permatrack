mkdir ../../data/mot17
cd ../../data/mot17
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip
rm MOT17.zip
mkdir annotations
mv MOT17/train .
mv MOT17/test .
rm -rf MOT17
cd ../../src/tools/
python convert_mot_to_coco.py
python interp_mot.py
python convert_mot_det_to_results.py