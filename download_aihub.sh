#!/bin/bash

curl -o "aihubshell" https://api.aihub.or.kr/api/aihubshell.do
chmod +x aihubshell
sudo cp aihubshell /usr/bin

aihubshell -mode d -datasetkey 71856 -aihubapikey 'ECEBABE0-71CE-493D-845B-350922EEEFCF'

# aihubshell -mode d -datasetkey 71856 -filekey 554058 -aihubapikey 'ECEBABE0-71CE-493D-845B-350922EEEFCF'

# aihubshell -mode d -datasetkey 71856 -filekey 554059 -aihubapikey 'ECEBABE0-71CE-493D-845B-350922EEEFCF'

mv '52.군 경계 작전 환경 합성데이터'/'3.개방데이터'/'1.데이터' aihub_military_synthetic

mv aihub_military_synthetic/Training/'01.원천데이터' aihub_military_synthetic/Training/images

mv aihub_military_synthetic/Training/'02.라벨링데이터' aihub_military_synthetic/Training/labels

mv aihub_military_synthetic/Validation/'01.원천데이터' aihub_military_synthetic/Validation/images

mv aihub_military_synthetic/Validation/'02.라벨링데이터' aihub_military_synthetic/Validation/labels

mkdir data/military_synthetic
mkdir data/military_synthetic/images
mkdir data/military_synthetic/annotations

