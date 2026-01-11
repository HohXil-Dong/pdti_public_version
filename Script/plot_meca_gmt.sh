#!/bin/zsh

INPUT_FILE=$1
if [ ! -s "$INPUT_FILE" ]; then
    echo "Usage: $0 <filename>"
    echo "Error: File '$INPUT_FILE' not found or is empty."
    exit 1
fi

gmt begin Focal_Mechanisms png E720

    gmt info "$INPUT_FILE" -I0.5/0.5 > tmp.region
    read -r R_VAL < tmp.region

    gmt coast $R_VAL -JM15c -Baf -BWSen+t"Focal Mechanisms" \
              -W0.5p,gray40 -G240/240/240 -Swhite

    # Plot Full Moment Tensor -Sm and Best Double Couple plane -Sd 
    gmt meca "$INPUT_FILE" -Sm0.4c -Gred -Ewhite -L0p
    gmt meca "$INPUT_FILE" -Sd0.4c -G- -E- -L0.15p,black

gmt end show

rm tmp.region
