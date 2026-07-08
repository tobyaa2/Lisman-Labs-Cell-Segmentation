# Lisman-Labs-Cell-Segmentation
Mostly AI coded cell segmentation and image processing model using SAM3 to process photos of Ciona embryos in vitro.

There are 2 parts to this Git, the Image Processor, and the SAM3 model. They can be used separately or together:

## Just SAM3
Follow the README.md in the "Meta SAM3" folder and install the SAM3 model.

Run: cd ../"Meta Sam3"
.venv/bin/python Sam3.py --image "/path/to/your_image.jpg"

## Just Image Processing
Run: 

python3 -m cell_counter.cli run \
    --filtered   "Example Images/c1_Filtered.png" \
    --unfiltered "Example Images/c1_Unfiltered.jpg" \
    --out        results/c1 \
    --save-overlay --save-csv

For image processing, you must provide a filtered and unfiltered version of an image. I used Cellpose for the filtering: https://github.com/mouseland/cellpose

Specifically, I used a modified version that allows you to save the filtered image. You can find the modified version here: https://drive.google.com/file/d/14XQAlQviPAPvP7Yb0W-7iM7N2DiFz53v/view?usp=sharing

To save a filtered image from Cellpose:
- Run: "../cellpose/bin/python" -m cellpose
- 

