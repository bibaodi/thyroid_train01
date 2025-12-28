#!/bin/path
cd images/
mkdir -p val && ls -1|grep "02_"|xargs -I '{}' mv '{}' ./val
mkdir -p train && mv ./*.png ./train/

cd ../labels
mkdir -p val && ls -1|grep "02_"|xargs -I '{}' mv '{}' ./val
mkdir -p train && mv ./*.txt ./train/

