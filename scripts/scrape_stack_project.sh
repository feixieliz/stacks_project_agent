#!/bin/bash

FILE_PATH="../stacks-project"

# get a list of files ending with .tex
for file in $(find $FILE_PATH -name "*.tex")
do
	cat $file >> all.tex
done


