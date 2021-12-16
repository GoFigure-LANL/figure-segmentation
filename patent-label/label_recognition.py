####
#
# Copyright 2020. Triad National Security, LLC. All rights reserved.
# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
# Department of Energy/National Nuclear Security Administration. All rights in the program are
# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting on its behalf a
# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
# others to do so.
#
####

## label recognition
## Authors: Ming Gong and Diane Oyen

# Standard packages
import argparse
import glob
import copy
import numpy as np
import re

# Image packages
from PIL import Image # Needed for reading TIF
import scipy.ndimage
from scipy.spatial import Delaunay
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import skimage
from skimage.io import imsave

# Text packages
import pytesseract


### Alpha-shape calculation with helper functions ###

def edge_length(point_a, point_b):
    """
    point_a and point_b are 2-tuples or lists of length 2, 
    representing endpoints of an edge. Returns the 
    Euclidean distance between point_a and point_b.
    """
    return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def add_alpha_edge(edges, i, j, only_outer=True):
    """
    Add a line between the i-th and j-th points,
    if not in the list already
    """
    if (i, j) in edges or (j, i) in edges:
        # already added
        assert (j, i) in edges, "Alpha-shape: Can't go twice over the same directed edge"
        if only_outer:
            # if both neighboring triangles are in shape, it's not a boundary edge
            edges.remove((j, i))
        return
    edges.add((i, j))


def alpha_shape(points, alpha, only_outer=True):
    assert points.shape[0] > 3, "Alpha-shape needs at least four points"

    tri = Delaunay(points)
    edges = set()

    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]

        a = edge_length(pa, pb)
        b = edge_length(pb, pc)
        c = edge_length(pc, pa)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c)) #overflow potential?
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_alpha_edge(edges, ia, ib, only_outer)
            add_alpha_edge(edges, ib, ic, only_outer)
            add_alpha_edge(edges, ic, ia, only_outer)
    return edges



### Helper functions ###

def draw_edges(points, edges, image_shape):
    edge_imarray = np.zeros(image_shape, dtype=np.uint8)
    for i,j in edges:
        row_i, col_i = points[i]
        row_j, col_j = points[j]
        rows, cols = skimage.draw.line(row_i, col_i, row_j, col_j)
        # or skimage.draw.line_aa(edge) for anti-aliased line
        edge_imarray[rows, cols] = 1
    return edge_imarray


def draw_mask(points, edges, image_shape):
    # Start with the polygon outlines
    mask_imarray = draw_edges(points, edges, image_shape)

    # Then fill them in
    mask_imarray = scipy.ndimage.binary_fill_holes(mask_imarray)
    return mask_imarray


def pad_box(bounding_box, padding, image_shape):
    """
    Add padding around given bounding box making sure not to exceed boundaries
    of the image. bounding_box is 4-tuple (min_row, min_col, max_row, max_col).
    Returns a new bounding_box as a 4-tuple.
    """
    (min_row, min_col, max_row, max_col) = bounding_box
    min_row = max(min_row - padding, 0)
    min_col = max(min_col - padding, 0)
    max_row = min(max_row + padding, image_shape[0])
    max_col = min(max_col + padding, image_shape[1])
    return (min_row, min_col, max_row, max_col)


def is_landscape(min_row, min_col, max_row, max_col):
    # Returns true if box is in "landscape" mode (wider than tall), and
    # false otherise.
    # Box should be rotated if it is taller than it is wide
    height = max_row - min_row
    width = max_col - min_col
    return width > height


def save_image(filename, bin_array):
    """
    Takes a binary array, inverts black/white, then saves to the given filename.
    """
    imsave(filename, skimage.img_as_ubyte(np.logical_not(bin_array)), check_contrast=False)
    return



### Label extraction function ###

def label_extraction(im, name, kernel_width,
                     region_size_min, region_size_max,
                     alpha, box_padding, output_dir,
                     regexp_filter=None,
                     save_regions=False, save_all_images=False):
    """
    Finds the figure labels in the given image. Returns a list of bounding boxes and text strings of the
    identified figure labels. Optionally, will save the images associated with the bounding box and/or 
    all images associated with intermediate image processing steps.
    """
    # Convert image to array
    orig_im = np.array(im)
    imarray = copy.deepcopy(orig_im)

    # If not grayscale, make it grayscale
    if len(imarray.shape) != 2:
        imarray = rgb2gray(imarray)

    # Threshold foreground/background
    threshold = threshold_otsu(imarray)
    img_bw = imarray < threshold
    if save_all_images:
        save_image(output_dir + "bw_" + name + ".png", img_bw)

    # Fill regions to segment
    img_fill = scipy.ndimage.binary_fill_holes(img_bw)
    #img_fill = np.array(img_fill, dtype=np.uint8)
    if save_all_images:
        save_image(output_dir + "fill_" + name + ".png", img_fill)

    # Find connected components
    # Use skimage instead of opencv for consistency and reduced package requirements
    pixels_labeled_by_component = skimage.measure.label(img_fill)
    regions_list = skimage.measure.regionprops(pixels_labeled_by_component)
    sizes = [x.area for x in regions_list]
    nb_components = len(regions_list)

    # Filter connected components based on size
    label_candidate_pixels = np.zeros((imarray.shape))
    for i in range(nb_components):
        if sizes[i] <= region_size_max and sizes[i] >= region_size_min:
            label_candidate_pixels[pixels_labeled_by_component == i + 1] = 1
    if save_all_images:
        save_image(output_dir + "candidate_" + name + ".png", label_candidate_pixels)
    # Check if any pixels are potentially text
    if not label_candidate_pixels.any():
        print("No text found in image " + name)
        return (None, None)

    # Erosion (dilation on inverse image) (dashed-line removal)
    kernel = np.ones((1,kernel_width), np.uint8) # horizontal kernel
    label_area = skimage.morphology.binary_erosion(label_candidate_pixels, kernel)
    kernel = np.ones((kernel_width,1), np.uint8) # vertical kernel
    label_area = skimage.morphology.binary_erosion(label_area, kernel)
    if save_all_images:
        save_image(output_dir + "dashgone_" + name + ".png", label_area)

    
    # Find alpha-shape (non-convex hull)
    points = np.transpose(np.where(label_area == 1))
    # Alpha-shape needs at least 4 points
    if not points.shape[0] > 3:
        print("No text found in image " + name)
        return (None, None)
    try:
        edges = alpha_shape(points, alpha=alpha, only_outer=True)
    except:
        print("Alpha shape failed for " + name)
        return (None, None)

    # Get mask of alpha-shapes
    edges_imarray = draw_edges(points, edges, label_area.shape)
    mask_imarray = draw_mask(points, edges, label_area.shape)
    if save_all_images:
        save_image(output_dir + "alphamask_" + name + ".png", mask_imarray)
        save_image(output_dir + "alpha_" + name + ".png", edges_imarray)

    # Get bounding boxes
    pixels_labeled_by_component = skimage.measure.label(mask_imarray)
    regions_list = skimage.measure.regionprops(pixels_labeled_by_component, orig_im)
    boxes = [] # list of results
    labels = []
    for i, region in enumerate(regions_list):
        bounding_box = region.bbox # (min_row, min_col, max_row, max_col)
        convex_mask = region.convex_image # binary image with same size as bounding_box
        # Check if there is any difference between the bounding box and the convex mask
        # i.e. noise pixels between the convex mask and the bounding box
        #region_image = region.intensity_image # not working as expected, so do it explicitly
        if box_padding > 0:
            bounding_box = pad_box(bounding_box, box_padding, imarray.shape)
        (min_row, min_col, max_row, max_col) = bounding_box
        region_image = skimage.img_as_ubyte(orig_im[min_row:max_row, min_col:max_col])
        

        # OCR the region by masking the whole image (OCR works better as full-page than on small image)
        # First check if it needs to be rotated
        if not is_landscape(min_row, min_col, max_row, max_col):
            region_image = skimage.transform.rotate(region_image, 270, resize=True)

        # Make blank image the size of the original image
        ocr_img = np.array(np.ones(imarray.shape) * np.amax(orig_im), dtype=np.uint8)
        # Copy region into blank image
        ocr_img[min_row:max_row, min_col:max_col] = np.array(orig_im[min_row:max_row, min_col:max_col], dtype=np.uint8)
        # Rotate the full-page, if needed
        if not is_landscape(min_row, min_col, max_row, max_col):
            ocr_img = skimage.transform.rotate(ocr_img, 270, resize=True)

        # Save region as its own image
        if (save_all_images):
            # Suppress "UserWarning: Low contrast image" because there is a lot
            # of whitespace in these images.
            imsave(output_dir + "region_unfiltered_" + name + "_" + str(i) + ".png",
                   skimage.img_as_ubyte(ocr_img), check_contrast=False)


        # Make sure input to tesseract is a good image format!
        text = pytesseract.image_to_string(skimage.img_as_ubyte(ocr_img))
        text = text.rstrip() # Remove any trailing whitespace
        
        # Filter based on recognized text
        if text == "":
            continue
        if regexp_filter is not None:
            if re.search(regexp_filter, text) is None:
                continue
            
        # Output per region: bounding box, text
        boxes.append(bounding_box)
        labels.append(text)

        # Save region as its own image
        if (save_regions):
            imsave(output_dir + "region_" + name + "_" + str(i) + ".png",
                   skimage.img_as_ubyte(region_image))

    return (boxes, labels)
            


### Main fundtion and commandline arguments ###

def get_args():
    """
    Defines commandline arguments.
    """
    parser = argparse.ArgumentParser(description='Read labels from patent figures.')
    parser.add_argument('--image_path', type=str, default='./',
                        help='All images in path will be processed [default=./]')
    parser.add_argument('--output', type=str, default='output',
                        help='Result table saved to [OUTPUT_DIR]/[OUTPUT].csv [default=output]')
    parser.add_argument('--save_regions', action='store_true', default=False,
                        help='If enabled, each region containing text will be saved to OUTPUT_DIR as an image')
    parser.add_argument('--save_all_images', action='store_true', default=False,
                        help='If enabled, all intermediate processing steps will be saved as images to OUTPUT_DIR')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Directory to store results in [default=./]')
    # Algorithm parameters
    parser.add_argument('--kernel_width', type=int, default=2,
                        help='Width, in number of pixels, of erosion kernel for dashed-line removal [default=2]')
    parser.add_argument('--region_size_min', type=int, default=45,
                        help='Size, in number of area pixels, for smallest candidate label region [default=45]')
    parser.add_argument('--region_size_max', type=int, default=5000,
                        help='Size, in number of area pixels, for largest candidate label region [default=5000]')
    parser.add_argument('--alpha', type=int, default=36,
                        help='Alpha-shape alpha parameter [default=36]')
    parser.add_argument('--box_padding', type=int, default=5,
                        help='Number of pixels for padding around bounding box [default=5]')
    parser.add_argument('--regexp_filter', type=str, default=None,
                        help='Filter out OCR text results if they do not contain a match to this regular expression [default is no filtering]. Example is \"[Ff][Ii][Gg]![0-9]\\.[0-9]\".')
            # --regexp_filter "[Ff][Ii][Gg]|[0-9]\.[0-9]" is used in SDU paper
    return parser.parse_args()


def main():
    args = get_args()
    img_fnames = glob.glob(args.image_path + '/*')
    image_names = []

    # Open output file for writing
    # Make sure output_dir has trailing '/'
    if args.output_dir == '':
        output_dir = './'
    else:
        output_dir = args.output_dir + '/'
    f = open(output_dir + args.output + '.csv', 'w')
    
    for fname in img_fnames:
        try:
            im = Image.open(fname)
        except:
            print("Failed to open image " + fname)
        image_name = fname.split('/')[-1] # strip leading path
        image_names.append(image_name)
        im_name = image_name.split('.')[0] # strip extension
        boxes, labels = label_extraction(im, im_name,
                                         args.kernel_width,
                                         args.region_size_min,
                                         args.region_size_max,
                                         args.alpha,
                                         args.box_padding,
                                         args.output_dir,
                                         args.regexp_filter,
                                         args.save_regions,
                                         args.save_all_images)

        # Write output
        if boxes is None or len(boxes) < 1:
            f.write(im_name + '\n')
        else:
            boxes = [repr(x) for x in boxes]
            labels = [repr(x) for x in labels]
            bbox_str = '"[' + ','.join(boxes) + ']"'
            label_str = '"[' + ','.join(labels) + ']"'
            f.write(','.join([im_name, bbox_str, label_str]))
            f.write('\n')

    f.close()

if __name__ == "__main__":
    # execute only if run as a script
    main()
