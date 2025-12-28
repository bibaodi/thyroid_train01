#!/bin/env bash

##1. source is src;
##2. target is train workspace dir;

for i in `find -P ../srcs/ -maxdepth 1 -name "*.pt" -type f`; do
	ln -s "${i}" ./
done
