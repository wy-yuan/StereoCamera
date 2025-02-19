from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def generate_chessboard_in_mm(
        size_mm=28,
        squares=8,
        dpi=600,
        light_color="white",
        dark_color="black",
        output_file="chessboard.png"
    ):

    # Convert mm to pixels
    size_in_pixels = int((size_mm / 25.4) * dpi)

    # Create the base image
    img = Image.new("RGB", (size_in_pixels, size_in_pixels), color=light_color)
    draw = ImageDraw.Draw(img)

    # Each square's size in pixels
    square_size = size_in_pixels // squares

    # Draw alternating squares
    for row in range(squares):
        for col in range(squares):
            if (row + col) % 2 == 1:  # alternate black/white squares
                top_left_x = col * square_size
                top_left_y = row * square_size
                bottom_right_x = (col + 1) * square_size
                bottom_right_y = (row + 1) * square_size
                draw.rectangle(
                    [top_left_x, top_left_y, bottom_right_x, bottom_right_y],
                    fill=dark_color
                )

    # Save the image with DPI metadata
    img.save(output_file, dpi=(dpi, dpi))
    print(f"Chessboard ({size_mm}mm x {size_mm}mm at {dpi} DPI) saved as {output_file}")

def generate_chessboard(rows, cols):
    # Create a chessboard pattern with alternating 0s and 1s
    chessboard = np.zeros((rows, cols))
    chessboard[1::2, ::2] = 1  # Fill 1s in alternating pattern
    chessboard[::2, 1::2] = 1
    plt.imshow(chessboard, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.savefig('chessboard_{}x{}.png'.format(rows, cols), bbox_inches='tight', pad_inches=0, dpi=1000)


if __name__ == "__main__":
    # Example usage: 28mm x 28mm board, 8 squares, 300 DPI
    # generate_chessboard_in_mm(
    #     size_mm=28,
    #     squares=8,
    #     dpi=100,
    #     light_color="white",
    #     dark_color="black",
    #     output_file="chessboard.png"
    # )

    # Generate an 8x7 chessboard pattern
    generate_chessboard(8, 7)