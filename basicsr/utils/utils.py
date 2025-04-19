import numpy as np

# filtering masks
def show_masks(masks, plt, alpha=0.7):
    if len(masks) == 0: return
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_masks:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [alpha]])
        img[m] = color_mask
    ax.imshow(img)
    return img

# filtering lbk masks
def show_lbk_masks(masks, plt, alpha=0.7):
    if len(masks) == 0: return
    img = np.ones((masks.shape[1], masks.shape[2], 1), dtype=np.uint8)
    mask_values = np.random.randint(1, 254, masks.shape[0])
    
    for ann, mask_value in zip(masks, mask_values):
        # print('ann statistical', np.unique(ann, return_counts=True))
        # color_mask = np.concatenate([np.random.random(3), [alpha]])
        img[ann] = mask_value
        # color = np.array([255, 255, 255])
        # h, w = ann.shape[-2:]
        # color_mask = ann.reshape(h, w, 1) * color_mask.reshape(1, 1, -1)
        # img += color_mask
    # ax.imshow(img)
    return img

def show_lbk_graymasks(masks, plt, alpha=0.7):
    if len(masks) == 0: return
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((masks.shape[1], masks.shape[2], 4))
    img[:, :, 3] = 0
    for ann in masks:
        # print('ann statistical', np.unique(ann, return_counts=True))
        color_mask = np.concatenate([np.random.random(3), [alpha]])
        img[ann] = color_mask
        # color = np.array([255, 255, 255])
        # h, w = ann.shape[-2:]
        # color_mask = ann.reshape(h, w, 1) * color_mask.reshape(1, 1, -1)
        # img += color_mask
    ax.imshow(img)
    return img

def show_largeRef_masks(masks, plt, alpha=0.7):
    if len(masks) == 0: return
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.zeros((masks.shape[1], masks.shape[2], 3))
    for ann in masks:
        color_mask = np.array([1., 1., 1.])
        img[ann] = color_mask
        # color = np.array([255, 255, 255])
        # h, w = ann.shape[-2:]
        # color_mask = ann.reshape(h, w, 1) * color_mask.reshape(1, 1, -1)
        # img += color_mask
    ax.imshow(img)
    return img

def show_mask(mask, ax, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

#375
def show_points(coords, labels, ax, marker_size=50):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, plt):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    