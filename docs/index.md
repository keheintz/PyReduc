---
layout: default
---

### Installing

For now, the project can only be downloaded directly from the GitHub repository. Then verify that you have the most recent versions imported in the setup.py file installed. 

When working on your own data it is important that the same folder structure is used. 

### Work-flow

The order of running the scripts are:

The script setup.py defines all the packages and gobal parameters. It is run by all the other scripts.


1) identify.py: identify arclines and fit a polynomium to the selected lines. 


2) extract_1d.py: extraction of the wavelength calibrated 1d spectrum. This produces as output the 1d science file nd a 1-sigma noise spectrum. There is both an optimally extracted version and a sum over an aperture.


3) standard.py and sensfunction.py: generation of the sensitivity function for flux calibration.


4) calibrate.py: use the output sensitivity function to calibrate the science spectrum (incl. the noise spectrum).


5) transform.py: We also have a preliminary script that can fit a 2d chebychef polynomium to the pixel to wavelength relation. Makes rectified 2d-spectrum as output. 


6) background.py: this is for background subtraction without having to use the trace of a source (similar to background in IRAF/twodspec).


### Code philosophy

Let's first make something quickly that works. Then we can refine it. Otherwise, I fear it will just go nowhere. We will assume that the science-file and arc-lamp file are made before running these reduction scripts. Scripts to produce these files will also be written (some of it is there : mkspecbias.py and mkspecflat.py). 

We have started from Yoonsoo Bach's notebook on github:
https://nbviewer.jupyter.org/github/ysBach/SNU_AOclass/blob/master/Notebooks/Spectroscopy_in_Python.ipynb?fbclid=IwAR22YsWpk-uNw7Iz9LGolRD6kbtpcTeqmYDKgfeRIQHQ42M8OLfRbRzJmeY

That we have used as a skeleton to start from.

Like iraf we try to keep output from scripts in the database folder.

### To-do

Missing (obviously a lot):
1) Ideally, identify.py could be made more userfriendly.

2) sensfunction.py needs to be improved to allow deletion of points during the fitting.

3) testing, testing, testing, debugging, debugging, debugging.

4) add more flexibility and user-control.

<!--- Text can be **bold**, _italic_, or ~~strikethrough~~. 
[Link to another page](./another-page.html). 
There should be whitespace between paragraphs. 
There should be whitespace between paragraphs. We recommend including a README, or a file with information about your project.
# Header 1
This is a normal paragraph following a header. GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.
## Header 2
> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.
### Header 3 --->
<!--- #### Header 4
*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.
##### Header 5
1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.
###### Header 6
| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |
### There's a horizontal rule below this.
* * *
### Here is an unordered list:
*   Item foo
*   Item bar
*   Item baz
*   Item zip
### And an ordered list:
1.  Item one
1.  Item two
1.  Item three
1.  Item four
### And a nested list:
- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item
### Small image
![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)
### Large image
![Branching](https://guides.github.com/activities/hello-world/branching.png)
### Definition lists can be used with HTML syntax.
<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl> 
--->
