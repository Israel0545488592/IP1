from ex1_utils import *

title_window = 'gamma correction'


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    global img
    img = imReadAndConvert(img_path, rep)

    def gamma_correct(val): cv.imshow(title_window, ((img / 255) ** (val / 50) * 255).astype(np.uint8))

    cv.namedWindow(title_window)
    cv.createTrackbar('gamma ', title_window, 50, 120, gamma_correct)
    cv.waitKey()


def main():
    gammaDisplay('bac_con.png', LOAD_RGB)


if __name__ == '__main__':
    main()
