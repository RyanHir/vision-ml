import math
import collections
import numpy as np
import tensorflow as tf



class Anchor(object):
    """Anchor class for anchor-based object detectors.

    Example:
    ```python
    anchor_boxes = Anchor(
        min_level=3,
        max_level=4,
        num_scales=2,
        aspect_ratios=[0.5, 1., 2.],
        anchor_size=4.,
        image_size=[256, 256],
    ).multilevel_boxes
    ```

    Attributes:
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added on each
        level. For instances, num_scales=2 adds one additional intermediate
        anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of float numbers representing the aspect ratio anchors
        added on each level. The number indicates the ratio of width to height.
        For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors on each
        scale level.
      anchor_size: float number representing the scale of size of the base
        anchor to the feature stride 2^level.
      image_size: a list of integer numbers or Tensors representing [height,
        width] of the input image size.
      multilevel_boxes: an OrderedDict from level to the generated anchor boxes of
        shape [height_l, width_l, num_anchors_per_location * 4].
      anchors_per_location: number of anchors per pixel location.
    """

    def __init__(
        self,
        min_level,
        max_level,
        num_scales,
        aspect_ratios,
        anchor_size,
        image_size,
    ):
      """Initializes the instance."""
      self.min_level = min_level
      self.max_level = max_level
      self.num_scales = num_scales
      self.aspect_ratios = aspect_ratios
      self.anchor_size = anchor_size
      self.image_size = image_size
      self.multilevel_boxes = self._generate_multilevel_boxes()

    def _generate_multilevel_boxes(self):
        """Generates multi-scale anchor boxes.

        Returns:
          An OrderedDict from level to anchor boxes of shape [height_l, width_l,
          num_anchors_per_location * 4].
        """
        multilevel_boxes = [] 
        for level in range(self.min_level, self.max_level + 1):
            boxes_l = []
            feat_size_y = math.ceil(self.image_size[0] / 2**level)
            feat_size_x = math.ceil(self.image_size[1] / 2**level)
            stride_y = tf.cast(self.image_size[0] / feat_size_y, tf.float32)
            stride_x = tf.cast(self.image_size[1] / feat_size_x, tf.float32)
            x = tf.range(stride_x / 2, self.image_size[1], stride_x)
            y = tf.range(stride_y / 2, self.image_size[0], stride_y)
            xv, yv = tf.meshgrid(x, y)
            for scale in range(self.num_scales):
                for aspect_ratio in self.aspect_ratios:
                    intermidate_scale = 2 ** (scale / self.num_scales)
                    base_anchor_size = self.anchor_size * 2**level * intermidate_scale
                    aspect_x = aspect_ratio**0.5
                    aspect_y = aspect_ratio**-0.5
                    half_anchor_size_x = base_anchor_size * aspect_x / 2.0
                    half_anchor_size_y = base_anchor_size * aspect_y / 2.0
                    # Tensor shape Nx4.
                    boxes = tf.stack(
                        [
                            yv - half_anchor_size_y,
                            xv - half_anchor_size_x,
                            yv + half_anchor_size_y,
                            xv + half_anchor_size_x,
                        ],
                        axis=-1,
                    )
                    boxes_l.append(boxes)
            # Concat anchors on the same level to tensor shape HxWx(Ax4).
            boxes_l = tf.concat(boxes_l, axis=-1)
            multilevel_boxes.append(boxes_l)
        return multilevel_boxes

    @property
    def anchors_per_location(self) -> int:
        return self.num_scales * len(self.aspect_ratios)


class GridAnchorGenerator:
    def __init__(self, box_spec_list, base_anchor_size, anchor_strides, anchor_offsets):
        self.base_anchor_size = base_anchor_size
        self.anchor_strides = anchor_strides
        self.anchor_offsets = anchor_offsets
        self.scales = []
        self.aspect_ratios = []
        self.box_specs = box_spec_list
        for spec in box_spec_list:
            scales, aspect_ratios = zip(*spec)
            self.scales.append(scales)
            self.aspect_ratios.append(aspect_ratios)
    def generate(self, feature_map_shape_list, im_height=1.0, im_width=1.0):
        im_height = tf.cast(im_height, tf.float32)
        im_width = tf.cast(im_width, tf.float32)

        if self.anchor_strides:
            anchor_strides = [(tf.cast(stride[0], tf.float32) / im_height,
                               tf.cast(stride[1], tf.float32) / im_width)
                              for stride in self.anchor_strides]
        else:
            anchor_strides = [(im_height / tf.cast(pair[0], tf.float32),
                               im_width / tf.cast(pair[1], tf.float32))
                              for pair in feature_map_shape_list]

        if self.anchor_offsets:
            anchor_offsets = [(tf.cast(offset[0], tf.float32) / im_height,
                               tf.cast(offset[1], tf.float32) / im_width)
                              for offset in self.anchor_offsets]
        else:
            anchor_offsets = [(0.5 * stride[0],
                               0.5 * stride[1])
                              for stride in anchor_strides]

        anchor_grid_list = []
        min_im_shape = tf.minimum(im_height, im_width)
        scale_height = min_im_shape / im_height
        scale_width = min_im_shape / im_width
        base_anchor_size = [scale_height * tf.constant(self.base_anchor_size[0], dtype=tf.float32),
                            scale_width  * tf.constant(self.base_anchor_size[1], dtype=tf.float32)]
        for feature_map_index, (grid_size, scales, aspect_ratios, stride, offset) \
            in enumerate(zip(feature_map_shape_list, self.scales, self.aspect_ratios, anchor_strides, anchor_offsets)):
            tiled_anchors = tile_anchors(
                    grid_height=grid_size[0],
                    grid_width=grid_size[1],
                    scales=scales,
                    aspect_ratios=aspect_ratios,
                    base_anchor_size=base_anchor_size,
                    anchor_stride=stride,
                    anchor_offset=offset)
            num_anchors_in_layer = tiled_anchors.shape[0]
            anchor_indices = feature_map_index * tf.ones([num_anchors_in_layer])
            tiled_anchors.feature_map_index = anchor_indices
            anchor_grid_list.append(tiled_anchors)
        return anchor_grid_list
    def num_anchors_per_location(self):
        """Returns the number of anchors per spatial location.
          Returns:
          a list of integers, one for each expected feature map to be passed to
          the Generate function.
        """
        return [len(box_specs) for box_specs in self.box_specs]

def tile_anchors(grid_height, grid_width, scales, aspect_ratios, base_anchor_size, anchor_stride, anchor_offset):
    ratio_sqrts = tf.sqrt(aspect_ratios)
    heights = scales / ratio_sqrts * base_anchor_size[0]
    widths = scales * ratio_sqrts * base_anchor_size[1]

    y_centers = tf.cast(tf.range(grid_height), tf.float32)
    x_centers = tf.cast(tf.range(grid_width), tf.float32)
    
    y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
    x_centers = x_centers * anchor_stride[1] + anchor_offset[1]

    x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

    widths_grid, x_centers_grid = tf.meshgrid(widths, x_centers)
    heights_grid, y_centers_grid = tf.meshgrid(heights, y_centers)

    bbox_sizes = tf.stack([heights_grid, widths_grid], axis=2)
    bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=2)
    
    bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
    bbox_centers = tf.reshape(bbox_centers, [-1, 2])
    
    bbox_corners = tf.concat([bbox_centers - 0.5 * bbox_sizes, bbox_centers + 0.5 * bbox_sizes], 1)
    return bbox_corners

def create_ssd_anchors(num_layers=6,
                       min_scale=0.2,
                       max_scale=1.5,
                       scales=None,
                       aspect_ratios=None,
                       aspect_ratios_last=None,
                       interpolated_scale_aspect_ratio=1.0,
                       base_anchor_size=None,
                       anchor_strides=None,
                       anchor_offsets=None,
                       reduce_boxes_in_lowest_layer=True):
    if base_anchor_size is None:
        base_anchor_size = [1.0, 1.0]
    box_specs_list = []
    if not scales:
        scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
                  for i in range(num_layers)]
    scales += [1.0]
    for layer, scale, scale_next in zip(range(num_layers), scales[:-1], scales[1:]):
        layer_box_specs = []
        if layer == 0 and reduce_boxes_in_lowest_layer:
            for aspect_ratio in aspect_ratios_last:
                layer_box_specs.append((scale, aspect_ratio))
        else:
            for aspect_ratio in aspect_ratios:
                layer_box_specs.append((scale, aspect_ratio))
            if interpolated_scale_aspect_ratio > 0.0:
                layer_box_specs.append((np.sqrt(scale * scale_next), interpolated_scale_aspect_ratio))
        box_specs_list.append(layer_box_specs)
    return GridAnchorGenerator(box_specs_list, base_anchor_size, anchor_strides, anchor_offsets)
