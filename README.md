# Lisman-Labs-Cell-Segmentation
Mostly AI-coded cell segmentation and image processing model using SAM3 to process photos of Ciona embryos in vitro.

There are 2 parts to this Git: the Image Processor and the SAM3 model. They can be used separately or together:

## Just SAM3
Download the "Meta SAM3" folder.

Follow the README.md in the "Meta SAM3" folder and install the SAM3 model.

You might need to apply for access to SAM3 [here](https://huggingface.co/facebook/sam3), then provide an API key from Hugging Face once you gain access. 

SAM3 is an exemplar segmentation model, which means that you draw boxes around examples of positives, and then it identifies similar objects based on the exemplar.

Run: cd ../"Meta Sam3"
.venv/bin/python SAM3.py --image "/path/to/your_image.jpg"

## Just Image Processing
Download the "Image Processor" folder.

Run: 

```
python3 -m cell_counter.cli run \
    --filtered   "Path to filtered/filtered_image.jpg" \
    --unfiltered "Path to unfiltered/unfiltered_image.jpg" \
    --out        results/"Image Folder" \
    --save-overlay --save-csv
```

For image processing, you must provide a filtered and unfiltered version of an image. I used [Cellpose](https://github.com/mouseland/cellpose) for the filtering.

Specifically, I used a modified version that allows you to save the filtered image.
- To install, download the modified version of Cellpose from [here](https://drive.google.com/file/d/1ktbT0AoZoflsC5j2ep9SAMwKxFNQc2Q-/view?usp=sharing)
- Follow the install instructions for your OS.


To save a filtered image from Cellpose:
- Run Cellpose
- In the Views window, change RGB to blue=B
- Go to File > Save Displayed RGB Image as png

## Both
Install everything from the Meta Sam3 and Image processing Steps.

Run:

```
python3 -m cell_counter.cli run \
    --filtered   "Path to filtered/filtered_image.jpg" \
    --unfiltered "Path to unfiltered/unfiltered_image.jpg" \
    --auto-exemplars \
    --out        results/"Image Folder" \
    --save-overlay --save-csv
```

Read the SAM3 README for instructions on using manual exemplars in the Image Processor.
