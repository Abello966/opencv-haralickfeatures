CC = g++
CFLAGS = -std=c++11 -L/usr/local/lib -I/usr/local/include/opencv -I/usr/local/include/opencv2
CEXTRA = -lopencv_shape -lopencv_stitching -lopencv_objdetect -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_core

haralick_feat: haralick_feat.cc
	$(CC) $(CFLAGS) $^ -g -o $@ -lm $(CEXTRA)
