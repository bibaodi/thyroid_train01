#!/bin/env bash
#eton@250312
read -p "remove all broken links in `pwd`  y/n"$'\n' ans
test ${ans} != 'y' && exit 0

find ./ -xtype l -delete

echo "done.."
