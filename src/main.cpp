#include <opencv2/opencv.hpp>

#include "config/config.h"
#include "config/manager/ConfigManager.h"

using namespace cv;

int main(int argc, char *argv[]) {

    ConfigManager::read();

    Mat3b img = imread(argv[1]);

    CFG_IMG_SIZE_Y = img.rows;
    CFG_IMG_SIZE_X = img.cols;

    int size = max(CFG_IMG_SIZE_Y, CFG_IMG_SIZE_X);

    Mat3b imgBG;
    resize( img, imgBG, Size(size, size) );
    boxFilter(imgBG, imgBG, -1, Size(CFG_FLT_SIZE, CFG_FLT_SIZE));
    boxFilter(imgBG, imgBG, -1, Size(CFG_FLT_SIZE, CFG_FLT_SIZE));
    imgBG = imgBG * CFG_BG_GAIN;

    Mat1f mask(CFG_IMG_SIZE_Y, CFG_IMG_SIZE_X, 1.0f);
    copyMakeBorder(mask, mask, (size-CFG_IMG_SIZE_Y)/2, (size-CFG_IMG_SIZE_Y)/2,
                             (size-CFG_IMG_SIZE_X)/2, (size-CFG_IMG_SIZE_X)/2, BORDER_CONSTANT, Scalar(0,0,0));
    copyMakeBorder(img, img, (size-CFG_IMG_SIZE_Y)/2, (size-CFG_IMG_SIZE_Y)/2,
                             (size-CFG_IMG_SIZE_X)/2, (size-CFG_IMG_SIZE_X)/2, BORDER_CONSTANT, Scalar(0,0,0));

    for (size_t i = 0; i < mask.total(); ++i) {
        img(i) = mask(i) * img(i) + (1.0f-mask(i)) * imgBG(i);
    }

    imwrite("out.png", img);

    ConfigManager::write();

    return 0;
}