import os
import collections

import tensorflow as tf
import six
import cv2
import numpy as np
from object_detection.utils import visualization_utils as viz_utils

print(tf.config.list_physical_devices())
os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.config.set_visible_devices([], "GPU")
model = tf.saved_model.load("saved_model")
# image = cv2.imread(
#     "comic_data/test/show-your-cool-best-favourite-one-piece-manga-panel-v0-giqm40q1voy91-ezgif.com-webp-to-jpg-converter.jpg"
# )
image = cv2.imread("comic_data/test/one-piece.jpg")
CATEGORY_INDEX = {0: {"id": 1, "name": "character"}}



def filter_out_boxes(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    keypoint_scores=None,
    keypoint_edges=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=0.5,
    agnostic_mode=False,
    line_thickness=4,
    mask_alpha=0.4,
    groundtruth_box_visualization_color="black",
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False,
):
    """Overlay labeled boxes on an image with formatted scores and label names.

    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.

    Args:
      image: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      instance_masks: a uint8 numpy array of shape [N, image_height, image_width],
        can be None.
      instance_boundaries: a numpy array of shape [N, image_height, image_width]
        with values ranging between 0 and 1, can be None.
      keypoints: a numpy array of shape [N, num_keypoints, 2], can
        be None.
      keypoint_scores: a numpy array of shape [N, num_keypoints], can be None.
      keypoint_edges: A list of tuples with keypoint indices that specify which
        keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
        edges from keypoint 0 to 1 and from keypoint 2 to 4.
      track_ids: a numpy array of shape [N] with unique track ids. If provided,
        color-coding of boxes will be determined by these ids, and not the class
        indices.
      use_normalized_coordinates: whether boxes is to be interpreted as
        normalized coordinates or not.
      max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
        all boxes.
      min_score_thresh: minimum score threshold for a box or keypoint to be
        visualized.
      agnostic_mode: boolean (default: False) controlling whether to evaluate in
        class-agnostic mode or not.  This mode will display scores but ignore
        classes.
      line_thickness: integer (default: 4) controlling line width of the boxes.
      mask_alpha: transparency value between 0 and 1 (default: 0.4).
      groundtruth_box_visualization_color: box color for visualizing groundtruth
        boxes
      skip_boxes: whether to skip the drawing of bounding boxes.
      skip_scores: whether to skip score when drawing a single detection
      skip_labels: whether to skip label when drawing a single detection
      skip_track_ids: whether to skip track id when drawing a single detection

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    box_to_keypoint_scores_map = collections.defaultdict(list)
    box_to_track_ids_map = {}
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(boxes.shape[0]):
        if max_boxes_to_draw == len(box_to_color_map):
            break
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if keypoint_scores is not None:
                box_to_keypoint_scores_map[box].extend(keypoint_scores[i])
            if track_ids is not None:
                box_to_track_ids_map[box] = track_ids[i]
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ""
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in six.viewkeys(category_index):
                            class_name = category_index[classes[i]]["name"]
                        else:
                            class_name = "N/A"
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = "{}%".format(round(100 * scores[i]))
                    else:
                        display_str = "{}: {}%".format(
                            display_str, round(100 * scores[i])
                        )
                if not skip_track_ids and track_ids is not None:
                    if not display_str:
                        display_str = "ID {}".format(track_ids[i])
                    else:
                        display_str = "{}: ID {}".format(display_str, track_ids[i])
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = "DarkOrange"
                elif track_ids is not None:
                    prime_multipler = viz_utils._get_multiplier_for_color_randomness()
                    box_to_color_map[box] = viz_utils.STANDARD_COLORS[
                        (prime_multipler * track_ids[i])
                        % len(viz_utils.STANDARD_COLORS)
                    ]
                else:
                    box_to_color_map[box] = viz_utils.STANDARD_COLORS[
                        classes[i] % len(viz_utils.STANDARD_COLORS)
                    ]
    
    return box_to_color_map, instance_masks, instance_boundaries

def get_detections(frame):
    """Get the detections boxes, scores and classes from the input frame"""

    # Convert the input array into a tensor
    # Create a batch then convert the input image values into unsigned ints ranging from 0 to 255
    input_tensor = np.expand_dims(frame, 0).astype(np.uint8)

    # Run the inference
    detections = model(input_tensor)

    # Get the results
    num_detections = int(detections.pop("num_detections"))
    # Create a dict from the results
    detections = {
        key: value[0, :num_detections].numpy() for key, value in detections.items()
    }
    # Convert the detection classes into te proper format (int)
    detections["detection_classes"] = detections["detection_classes"].astype(np.uint8)
    detections["num_detections"] = num_detections

    return detections


def write_visualizations(image, boxes, classes, scores):
    """Write the detections results on a frame"""

    # Visualize the results then draw the output boxes and classes on images
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        CATEGORY_INDEX,
        use_normalized_coordinates=True,
        max_boxes_to_draw=30,
        min_score_thresh=0.3,
        agnostic_mode=False,
        line_thickness=5,
    )

    return image

def get_filtered(image, boxes, classes, scores):
    return filter_out_boxes(
        image,
        boxes,
        classes,
        scores,
        CATEGORY_INDEX,
        use_normalized_coordinates=True,
        max_boxes_to_draw=30,
        min_score_thresh=0.3,
        agnostic_mode=False,
        line_thickness=5,
    )


# yolobboxes = model(
#     np.expand_dims(a, axis=0)
# )  # detection_multiclass_scores (1, 100, 2), detection_anchor_indices (1, 100), num_detections (1,), raw_detection_scores (1, 51150, 2), raw_detection_boxes (1, 51150, 4), detection_scores (1, 100), detection_classes (1, 100), detection_boxes (1, 100, 4)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)
image = np.pad(
    image, ((50, 50), (50, 50), (0, 0)), mode="constant", constant_values=255
)

detections = get_detections(image)
filtered_box_to_color_map, filtered_instance_masks, filtered_instance_boundaries = get_filtered(
    image,
    detections["detection_boxes"],
    detections["detection_classes"],
    detections["detection_scores"],
)

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
for box, color in filtered_box_to_color_map.items():
    h_iamge, w_image, _ = image.shape
    ymin, xmin, ymax, xmax = box
    ymin, xmin, ymax, xmax = int(ymin * h_iamge), int(xmin * w_image), int(ymax * h_iamge), int(xmax * w_image)
    cv2.rectangle(image, (xmin,ymin),(xmax,ymax), color=(255, 0, 0))
print(filtered_box_to_color_map, filtered_instance_masks, filtered_instance_boundaries)

cv2.imshow("d", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
