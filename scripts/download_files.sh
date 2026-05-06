#!/usr/bin/bash

BaseUrl="https://physionet.org/files/eegmmidb/1.0.0"
OutputDir="../data/raw/"

#download files using loop 
for i in $(seq -f "%03g" 1 20);do
    for j in {04,08,12};do
        filename="S${i}R${j}.edf"
        url="${BaseUrl}/S${i}/${filename}"
        echo "Downloading ${filename} from ${url}"
        wget -O "${OutputDir}/${filename}" "${url}"
        wget -O "${OutputDir}/${filename%.edf}.event" "${url%.edf}.event"
    done
done
