import numpy as np
from PIL import Image

class QuadTreeNode:
    def __init__(self, x, y, size, color=None):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0

def calculate_average_color(image, x, y, size):
    sub_image = image[y:y+size, x:x+size]
    avg_color = np.mean(sub_image, axis=(0, 1)).astype(int)
    return tuple(avg_color)

def color_difference(color1, color2):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))

def build_quadtree(image, x, y, size, threshold):
    if size <= 1:
        return QuadTreeNode(x, y, size, color=tuple(image[y, x]))

    avg_color = calculate_average_color(image, x, y, size)
    max_diff = 0
    for dy in range(size):
        for dx in range(size):
            diff = color_difference(avg_color, tuple(image[y + dy, x + dx]))
            if diff > max_diff:
                max_diff = diff

    if max_diff <= threshold:
        return QuadTreeNode(x, y, size, color=avg_color)

    half_size = size // 2
    node = QuadTreeNode(x, y, size)
    node.children.append(build_quadtree(image, x, y, half_size, threshold))
    node.children.append(build_quadtree(image, x + half_size, y, half_size, threshold))
    node.children.append(build_quadtree(image, x, y + half_size, half_size, threshold))
    node.children.append(build_quadtree(image, x + half_size, y + half_size, half_size, threshold))
    return node

def draw_quadtree(image, node):
    if node.is_leaf():
        image[node.y:node.y + node.size, node.x:node.x + node.size] = node.color
    else:
        for child in node.children:
            draw_quadtree(image, child)

def compress_image(input_path, output_path, threshold):
    image = Image.open(input_path)
    image = image.convert('RGB')
    image_np = np.array(image)

    height, width, _ = image_np.shape
    size = 1 << max(height, width).bit_length()

    padded_image = np.zeros((size, size, 3), dtype=np.uint8)
    padded_image[:height, :width, :] = image_np

    quadtree = build_quadtree(padded_image, 0, 0, size, threshold)

    compressed_image = np.zeros_like(padded_image)
    draw_quadtree(compressed_image, quadtree)

    compressed_image = compressed_image[:height, :width, :]
    result_image = Image.fromarray(compressed_image)
    result_image.save(output_path)

if __name__ == "__main__":
    input_path = 'input_image.jpg'
    output_path = 'compressed_image.jpg'
    threshold = 30

    compress_image(input_path, output_path, threshold)
