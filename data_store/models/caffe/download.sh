#!/bin/bash
file_bz=shape_predictor_68_face_landmarks.dat.bz2
file_dat=shape_predictor_68_face_landmarks.dat


if [ ! -f "$file_dat" ];then

	if [  -f "$file_bz" ];then
		echo "Unzipping file $file_bz"
		bzip2 -d $file_bz
	elif [ ! -f "$file_dat" ];then
		echo "Downloading file $file_bz"
		wget http://dlib.net/files/$file_bz
	else
		echo "Erro: $file_bz not found"
		bzip2 -d $file_bz
	fi
else
	echo "File already exists"
fi
echo "Done"