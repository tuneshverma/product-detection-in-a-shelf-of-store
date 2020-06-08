
---
The model is based on pytorch.
Install the Pytorch according to the system and nvidia toolkit.

Please find image2products.json file with image name and number of product in it. The dict containing an entry for every shelf image in the test set with
image name as �key� and number of products present in it as �value� .

metrics.json  for the mAP,  precision and recall. Actually I could not find annotation for the test images so I divide the train in two sets train and validation set. Therefore the mAP,  precision and recall are calucated using the validation set.

#Data Preperation

I divide the annotation.txt file to annotation_train.txt and annotation_test.txt file. 
Using create_data_lists_grocerydataset.py file I created TRAIN_images.json, TEST_images.json and TRAIN_objects.json, TEST_objects.json containing images path and their labels respectively.


---

#Anchor boxes(prior boxes)

**Priors are precalculated, fixed boxes which collectively represent this universe of probable and approximate box predictions**.

Priors are manually but carefully chosen based on the shapes and sizes of ground truth objects in our dataset. By placing these priors at every possible location in a feature map, we also account for variety in position.

In defining the priors, the authors specify that –

- **they will be applied to various low-level and high-level feature maps**, viz. those from `conv4_3`, `conv7`, `conv8_2`, `conv9_2`, `conv10_2`, and `conv11_2`. 

According to the SSD paper.

- **if a prior has a scale `s`, then its area is equal to that of a square with side `s`**. The largest feature map, `conv4_3`, will have priors with a scale of `0.1`, i.e. `10%` of image's dimensions, while the rest have priors with scales linearly increasing from `0.2` to `0.9`. As you can see, larger feature maps have priors with smaller scales and are therefore ideal for detecting smaller objects.

- **At _each_ position on a feature map, there will be priors of various aspect ratios**. All feature maps will have priors with ratios `1:1, 2:1, 1:2`. The intermediate feature maps of `conv7`, `conv8_2`, and `conv9_2` will _also_ have priors with ratios `3:1, 1:3`. Moreover, all feature maps will have *one extra prior* with an aspect ratio of `1:1` and at a scale that is the geometric mean of the scales of the current and subsequent feature map.

| Feature Map From | Feature Map Dimensions | Prior Scale | Aspect Ratios | Number of Priors per Position | Total Number of Priors on this Feature Map |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| `conv4_3`      | 38, 38       | 0.1 | 1:1, 2:1, 1:2 + an extra prior | 4 | 5776 |
| `conv7`      | 19, 19       | 0.2 | 1:1, 2:1, 1:2, 3:1, 1:3 + an extra prior | 6 | 2166 |
| `conv8_2`      | 10, 10       | 0.375 | 1:1, 2:1, 1:2, 3:1, 1:3 + an extra prior | 6 | 600 |
| `conv9_2`      | 5, 5       | 0.55 | 1:1, 2:1, 1:2, 3:1, 1:3 + an extra prior | 6 | 150 |
| `conv10_2`      | 3,  3       | 0.725 | 1:1, 2:1, 1:2 + an extra prior | 4 | 36 |
| `conv11_2`      | 1, 1       | 0.9 | 1:1, 2:1, 1:2 + an extra prior | 4 | 4 |
| **Grand Total**      |    –    | – | – | – | **8732 priors** |

There are a total of 8732 priors defined for the SSD300!


According to the problem statement given to implement a single shot object
detector with only �one� anchor box per feature-map cell .

I made the required changes.

| Feature Map From | Feature Map Dimensions | Prior Scale | Aspect Ratios | Number of Priors per Position | Total Number of Priors on this Feature Map |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| `conv4_3`      | 38, 38       | 2:1 | 1 | 1444 |
| `conv7`      | 19, 19       | 2:1| 1 | 361 |
| `conv8_2`      | 10, 10       | 2:1 | 1 | 100 |
| `conv9_2`      | 5, 5       | 2:1| 1 | 25 |
| `conv10_2`      | 3,  3       | 2:1| 1 | 9 |
| `conv11_2`      | 1, 1       | 2:1 | 1 | 1 |
| **Grand Total**      |    –    | – | – | – | **1940 priors** |

So it was asked to inplement the detector with only one anchor box(prior box) per feature map cell. Accordingly I made the changes and as shown in the table above the total number of anchor boxes are 1940.

The changes I made can be seen in the function 'create_prior_boxes' and in class 'PredictionConvolutions' in 'model.py' file


#### Data Transforms

See 'transform()' in [`utils.py`]

This function applies the following transformations to the images and the objects in them –

- Randomly **adjust brightness, contrast, saturation, and hue**, each with a 50% chance and in random order.

- With a 50% chance, **perform a _zoom out_ operation** on the image. This helps with learning to detect small objects. The zoomed out image must be between `1` and `4` times as large as the original. The surrounding space could be filled with the mean of the ImageNet data.

- Randomly crop image, i.e. **perform a _zoom in_ operation.** This helps with learning to detect large or partial objects. Some objects may even be cut out entirely. Crop dimensions are to be between `0.3` and `1` times the original dimensions. The aspect ratio is to be between `0.5` and `2`. Each crop is made such that there is at least one bounding box remaining that has a Jaccard overlap of either `0`, `0.1`, `0.3`, `0.5`, `0.7`, or `0.9`, randomly chosen, with the cropped image. In addition, any bounding boxes remaining whose centers are no longer in the image as a result of the crop are discarded. There is also a chance that the image is not cropped at all.

- With a 50% chance, **horizontally flip** the image.

- **Resize** the image to `300, 300` pixels. This is a requirement of the SSD300.

- Convert all boxes from **absolute to fractional boundary coordinates.** At all stages in our model, all boundary and center-size coordinates will be in their fractional forms.

- **Normalize** the image with the mean and standard deviation of the ImageNet data that was used to pretrain our VGG base.

As mentioned in the paper, these transformations play a crucial role in obtaining the stated results.

### Base Convolutions

See `VGGBase` in [`model.py`]

Here, we **create and apply base convolutions.**

The layers are initialized with parameters from a pretrained VGG-16 with the `load_pretrained_layers()` method.

We're especially interested in the lower-level feature maps that result from `conv4_3` and `conv7`, which we return for use in subsequent stages.

### Auxiliary Convolutions

See `AuxiliaryConvolutions` in [`model.py`]

Here, we **create and apply auxiliary convolutions.**

Use a [uniform Xavier initialization](https://pytorch.org/docs/stable/nn.html#torch.nn.init.xavier_uniform_) for the parameters of these layers.

We're especially interested in the higher-level feature maps that result from `conv8_2`, `conv9_2`, `conv10_2` and `conv11_2`, which we return for use in subsequent stages.

### Prediction Convolutions

See `PredictionConvolutions` in [`model.py`]
Here, we **create and apply localization and class prediction convolutions** to the feature maps from `conv4_3`, `conv7`, `conv8_2`, `conv9_2`, `conv10_2` and `conv11_2`.

These layers are initialized in a manner similar to the auxiliary convolutions.

We also **reshape the resulting prediction maps and stack them** as discussed. Note that reshaping in PyTorch is only possible if the original tensor is stored in a [contiguous](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.contiguous) chunk of memory.

As expected, the stacked localization and class predictions will be of dimensions `1940, 4` and `1940, 1` respectively.

### Putting it all together

See `SSD300` in [`model.py`]

Here, the **base, auxiliary, and prediction convolutions are combined** to form the SSD.

There is a small detail here – the lowest level features, i.e. those from `conv4_3`, are expected to be on a significantly different numerical scale compared to its higher-level counterparts. Therefore, the authors recommend L2-normalizing and then rescaling _each_ of its channels by a learnable value.

### Priors

See `create_prior_boxes()` under `SSD300` in [`model.py`]

This function **creates the priors in center-size coordinates** as defined for the feature maps from `conv4_3`, `conv7`, `conv8_2`, `conv9_2`, `conv10_2` and `conv11_2`, _in that order_. Furthermore, for each feature map, we create the priors at each tile by traversing it row-wise.

This ordering of the 1940 priors thus obtained is very important because it needs to match the order of the stacked predictions.

### Multibox Loss

See `MultiBoxLoss` in [`model.py`]

Two empty tensors are created to store localization and class prediction targets, i.e. _ground truths_, for the 1940 predicted boxes in each image.

We **find the ground truth object with the maximum Jaccard overlap for each prior**, which is stored in `object_for_each_prior`.

We want to avoid the rare situation where not all of the ground truth objects have been matched. Therefore, we also **find the prior with the maximum overlap for each ground truth object**, stored in `prior_for_each_object`. We explicitly add these matches to `object_for_each_prior` and artificially set their overlaps to a value above the threshold so they are not eliminated.

Based on the matches in `object_for_each prior`, we set the corresponding labels, i.e. **targets for class prediction**, to each of the 1940 priors. For those priors that don't overlap significantly with their matched objects, the label is set to _background_.

Also, we encode the coordinates of the 1940 matched objects in `object_for_each prior` in offset form `(g_c_x, g_c_y, g_w, g_h)` with respect to these priors, to form the **targets for localization**. Not all of these 1940 localization targets are meaningful. As we discussed earlier, only the predictions arising from the non-background priors will be regressed to their targets.

The **localization loss** is the [Smooth L1 loss](https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss) over the positive matches.

Perform Hard Negative Mining – rank class predictions matched to _background_, i.e. negative matches, by their individual [Cross Entropy losses](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss). The **confidence loss** is the Cross Entropy loss over the positive matches and the hardest negative matches. Nevertheless, it is averaged only by the number of positive matches.

The **Multibox Loss is the aggregate of these two losses**, combined in the ratio `α`. In our case, they are simply being added because `α = 1`.

#Training Parameters

batch_size = 16  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-4  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate

I used Adam optimizer instread of SGD

# Evaluation

See [`eval.py`]

The data-loading and checkpoint parameters for evaluating the model are at the beginning of the file, so you can easily check or modify them should you wish to.

To begin evaluation, run eval.py file.

Evaluation has to be done at a `min_score` of `0.18`, an NMS `max_overlap` of `0.25`, and `top_k` of `200` to allow fair comparision of results with the paper and other implementations.

**Parsed predictions are evaluated against the ground truth objects.** The evaluation metric is the _Mean Average Precision (mAP)_.

We will use `calculate_mAP()` in [`utils.py`] for this purpose.


# Inference

See [`detect.py`]

Run:
python detect.py --imagepath [image path]

Point to the model you want to use for inference with the `checkpoint` parameter at the beginning of the code.

Then, you can use the `detect()` function to identify and visualize objects in an RGB image.

This function first **preprocesses the image by resizing and normalizing its RGB channels** as required by the model. It then **obtains raw predictions from the model, which are parsed** by the `detect_objects()` method in the model. The parsed results are converted from fractional to absolute boundary coordinates, their labels are decoded with the `label_map`, and they are **visualized on the image**.

What is the purpose of using multiple anchors per feature map cell?

Answer - In SSD algorithim we get some feature maps from low and high level after appying convolution layers. Multiple anchor per feature map cell are boxes of different aspect ratio. As we know in objects detection problem an image can have multiple objects of different size and shape. To predict bounding boxes for objects of different sizes and shape from a feature map cell we define some box with different aspect ratio. We calulate IOU from ground truth boxes and all the aspect ratio boxes and chose box with highest IOU and confidence score.
 
Does this problem require multiple anchors? Please justify your answer.

Answer - We don't require multiple anchor for this problem because here we have only one object class which have almost same aspect ratio through out the dataset. So single box of certain aspect ratio from all the different feature map cell can be used to for training.
