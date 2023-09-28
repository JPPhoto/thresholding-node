# thresholding-node
The thresholding node takes in a source image and outputs three mask images representing the portions of the source
that are highlights, midtones, and shadows, per the brightness values (0-255) that you set.

Example input:
![Bee Robot](input.png)

Highlights output:
![Bee Robot Highlights](highlights.png)

Midtones output:
![Bee Robot Midtones](midtones.png)

Shadows output:
![Bee Robot Shadows](shadows.png)

This node also can blur the lookup table that it uses to separate the three brightness regions, resulting in a smoother mask. Note that this is not the same as blurring the masks themselves.

Highlights output w/LUT blur:
![Bee Robot Highlights, LUT Blur](highlights_lutblur.png)

Midtones output w/LUT blur:
![Bee Robot Midtones](midtones_lutblur.png)

Shadows output w/LUT blur:
![Bee Robot Shadows](shadows_lutblur.png)
