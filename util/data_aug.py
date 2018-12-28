import cv2, math, random
import numpy as np

def py_scale(img, label=None, scalar=1.3):

    scalar = random.uniform(1.0, scalar)

    h = int(img.shape[0] * scalar)
    w = int(img.shape[1] * scalar)

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    if label:
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        return img, label

    else:
        return img

def py_rotate_raw(img, angle):
    if img.shape[0] >= img.shape[1]:
        long = img.shape[0]
        short = img.shape[1]
    else:
        long = img.shape[1]
        short = img.shape[0]
    theta = math.atan(long / short) * 180 / math.pi
    k = math.cos(theta * math.pi / 180) / math.cos(abs(theta - abs(angle)) * math.pi / 180)
    mat = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1 / k)
    img = cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))

    return img


def RandomHorizontalFlip(img, label=None):
    pb = random.uniform(0.0, 1.0)
    if pb > 0.5:
        img = cv2.flip(img, 1)
        if label:
            label = cv2.flip(label, 1)
            assert label.dtype == np.uint8

    if label:
        return img, label
    else:
        return img

def RandomResizeCrop(img, label=None, scalar=1.3, size=None):
    if size:
        h, w = img.shape[0:2]
        h_t, w_t = size
        if h < h_t or w < w_t:
            img = np.pad(img,((0, max(0,h_t - h)),(0, max(0, w_t - w)), (0,0)), "constant",constant_values=0)



    h_t, w_t = size if size else img.shape[0:2]

    if label:
        img, label = py_scale(img, label=label, scalar=scalar)
    else:
        img = py_scale(img, label=None, scalar=scalar)

    h_big, w_big = img.shape[0:2]

    off_h = random.randint(0, h_big - h_t)
    off_w = random.randint(0, w_big - w_t)

    img = img[off_h: off_h + h_t, off_w: off_w + w_t]
    assert img.dtype == np.uint8

    if label:
        label = label[off_h: off_h + h_t, off_w: off_w + w_t]
        assert label.dtype == np.uint8
        return img, label
    else:
        return img


def RandomRotate(img, label=None, angle=0):
    angle = random.uniform(0, angle)
    img = py_rotate_raw(img, angle)
    assert img.dtype == np.uint8
    if label:
        label = py_rotate_raw(label, angle)
        return img, label
    else:
        return img

def HSV_jitter(img, min_v, max_v, which):
    assert which in ["H", "S", "V"]
    dic = {'H': 0, 'S': 1, 'V': 2}

    cimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_v = random.uniform(min_v, max_v)
    # print(hsv_v)
    which_c = cimg[:, :, dic[str(which)]]
    which_c *= hsv_v
    if which == 'H':
        which_c = np.clip(which_c, a_min=0, a_max=180)
    else:
        which_c = np.clip(which_c, a_min=0, a_max=255)

    cimg[:, :, dic[str(which)]] = which_c
    cimg = cimg.astype(np.uint8)
    return cv2.cvtColor(cimg, cv2.COLOR_HSV2BGR)




if __name__ == "__main__":

    import numpy as np
    import cv2

    img = np.random.randint(0, 255, size=[128,128,3]).astype(np.uint8)
    cv2.imshow("img1", img)
    cv2.waitKey(1000)

    img = RandomHorizontalFlip(img)
    print("flip", img.shape)
    img = RandomResizeCrop(img, scalar=1.3, size=(128, 128))
    print("sizecrop", img.shape)
    img = HSV_jitter(img, min_v=0.8, max_v=1.2, which="H")
    print("hsv", img.shape)
    img = RandomRotate(img, angle=45)
    print("rorate", img.shape)

    cv2.imshow("img2", img)
    cv2.waitKey()
