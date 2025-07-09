#!/bin/bash

curl -o "aihubshell" https://api.aihub.or.kr/api/aihubshell.do
chmod +x aihubshell
sudo cp aihubshell /usr/bin

aihubshell -mode d -datasetkey 71856 -aihubapikey 'ECEBABE0-71CE-493D-845B-350922EEEFCF'

# aihubshell -mode d -datasetkey 71856 -filekey 554058 -aihubapikey 'ECEBABE0-71CE-493D-845B-350922EEEFCF'

# aihubshell -mode d -datasetkey 71856 -filekey 554059 -aihubapikey 'ECEBABE0-71CE-493D-845B-350922EEEFCF'

mv 52.군_경계_작전_환경_합성데이터/3.개방데이터/1.데이터/ aihub_military_synthetic
# rm -rf 52.군_경계_작전_환경_합성데이터
mv aihub_military_synthetic/Training/01.원천데이터 aihub_military_synthetic/Training/images

mv aihub_military_synthetic/Training/02.라벨링데이터 aihub_military_synthetic/Training/labels

mv aihub_military_synthetic/Validation/01.원천데이터 aihub_military_synthetic/Validation/images

mv aihub_military_synthetic/Validation/02.라벨링데이터 aihub_military_synthetic/Validation/labels

mkdir data/military_synthetic
mkdir data/military_synthetic/images
mkdir data/military_synthetic/annotations

mv aihub_military_synthetic/Training/images data/military_synthetic/images/train
mv aihub_military_synthetic/Validation/images data/military_synthetic/images/val


cd aihub_military_synthetic/Training/labels
unzip TL_IR_SU_NT.zip
unzip TL_EO_WI_DT.zip
unzip TL_EO_SU_NT.zip
unzip TL_EO_SU_DT.zip
# rm ./*.zip
cd ../../..

cd aihub_military_synthetic/Validation/labels
unzip VL_IR_SU_NT.zip
unzip VL_EO_WI_DT.zip
unzip VL_EO_SU_NT.zip
unzip VL_EO_SU_DT.zip
# rm ./*.zip
cd ../../..

cd data/military_synthetic/images/train
unzip TS_IR_SU_NT.zip
unzip TS_EO_WI_DT.zip
unzip TS_EO_SU_NT.zip
unzip TS_EO_SU_DT.zip
# rm ./*.zip
cd ../../../..

cd data/military_synthetic/images/val
unzip VS_IR_SU_NT.zip
unzip VS_EO_WI_DT.zip
unzip VS_EO_SU_NT.zip
unzip VS_EO_SU_DT.zip
# rm ./*.zip
cd ../../../..

python combine_aihub_jsons.py