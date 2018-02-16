CC = g++

CFLAGS += \
    -std=c++11 \
    -I/usr/local/include/opencv \
    -I/usr/local/include/opencv2

LDFLAGS += \
    -L/usr/local/lib

LIBS += \
    -lopencv_shape \
    -lopencv_stitching \
    -lopencv_objdetect \
    -lopencv_superres \
    -lopencv_videostab \
    -lopencv_calib3d \
    -lopencv_features2d \
    -lopencv_highgui \
    -lopencv_videoio \
    -lopencv_imgcodecs \
    -lopencv_video \
    -lopencv_photo \
    -lopencv_ml \
    -lopencv_imgproc \
    -lopencv_flann \
    -lopencv_core \
    -lm

haralick_feat: src/haralick_feat.cc
	$(CC) $(CFLAGS) $^ -g -o $@ $(LDFLAGS) $(LIBS)
