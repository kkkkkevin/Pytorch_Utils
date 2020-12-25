

def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(
                round(
                    max_size *
                    min_original_size /
                    max_original_size))
    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return oh, ow


def get_size(image_size, size, max_size=None):
    if isinstance(size, (list, tuple)):
        return size[::-1]
    else:
        return get_size_with_aspect_ratio(image_size, size, max_size)
