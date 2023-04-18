from ex1_utils import *

title_window = 'gamma correction'
RESOLUTION = 40


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    global img
    global last_val
    img = imReadAndConvert(img_path, rep)
    last_val = RESOLUTION

    def gamma_correct(val):

        global img
        global last_val
        val += 1

        img **= val / last_val
        cv.imshow(title_window, discrete_normelize(img))
        last_val = val

    cv.namedWindow(title_window)
    cv.createTrackbar('gamma ', title_window, RESOLUTION, RESOLUTION * 3, gamma_correct)
    cv.waitKey()


if __name__ == '__main__':  gammaDisplay('bac_con.png', LOAD_RGB)