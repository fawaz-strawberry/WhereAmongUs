@echo off
set input_image=%1
set output_masks_dir=../../Output_Images/

REM Extract the input image filename without the extension
for %%A in ("%input_image%") do (
    set "filename=%%~nA"
)

REM Set the output_masks_dir to the correct subfolder
set output_masks_dir=%output_masks_dir%%filename%

REM Run the segmentation script
python C:/Users/fawaz/Documents/Github/WhereAmongUs/Scripts/OverlayGenerator/quickRun.py %input_image%

REM Run the mask comparison script
python C:/Users/fawaz/Documents/Github/WhereAmongUs/Scripts/OverlayGenerator/main_program.py %input_image% %output_masks_dir%
