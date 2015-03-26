#include <iostream>
#include <fstream>	
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

using namespace std;
using namespace cv;

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

void CannyThreshold(int, void*)
{
	/// Reduce noise with a kernel 3x3
	blur(src_gray, detected_edges, Size(3, 3));

	/// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);

	src.copyTo(dst, detected_edges);
	imshow(window_name, dst);
}
int sliderExample(){
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;
		/// Load an image
		while (true){

			cap.read(src);
		
			//cap >> src;

			if (!src.data)
			{
				return -1;
			}

			/// Create a matrix of the same type and size as src (for dst)
			dst.create(src.size(), src.type());

			/// Convert the image to grayscale
			cvtColor(src, src_gray, CV_BGR2GRAY);

			/// Create a window
			namedWindow(window_name, CV_WINDOW_AUTOSIZE);

			/// Create a Trackbar for user to enter threshold
			createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

			/// Show the image	
			CannyThreshold(0, 0);

			waitKey(5);
		}
}
int thresholdExample(){
	VideoCapture cap(0); // open the default camera
	
	if (!cap.isOpened())  // check if we succeeded
		return -1;
	/// Load an image
	cap >> src;
	while (true){
			
		cap.read(src);
		if (!src.data)
		{
			return -1;
		}
		cvtColor(src, src, CV_RGB2GRAY);
		// Apply adaptive threshold.
		adaptiveThreshold(src, src, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 3, 5);
		// Attempt to sharpen the image.
		GaussianBlur(src, src, cv::Size(0, 0), 3);
		addWeighted(src, 1.5, src, -0.5, 0, src);

		/// Create a window
		namedWindow(window_name, CV_WINDOW_AUTOSIZE);
		imshow(window_name, src);
		waitKey(3);
	}
}
int OCRExample(){
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;
	/// Load an image
	while (true){
		waitKey();
		cap.read(src);
		//cap >> src;

		if (!src.data)
		{
			return -1;
		}

		/// Create a matrix of the same type and size as src (for dst)
		dst.create(src.size(), src.type());

		/// Convert the image to grayscale
		cvtColor(src, src_gray, CV_BGR2GRAY);

		/// Create a window
		namedWindow(window_name, CV_WINDOW_AUTOSIZE);

		/// Create a Trackbar for user to enter threshold
		createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

		/// Show the image	
		CannyThreshold(0, 0);
		
		//imwrite("../Tesseract-OCR/test.jpg", dst);
		//cout << system("cd ../Tesseract-OCR && tesseract.exe test.jpg output && more output.txt");
		
	}

}
int webcamExample(){
	cvNamedWindow("Camera_Output", 1);    //Create window
	CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);  //Capture using any camera connected to your system
	while (1){ //Create infinte loop for live streaming
		IplImage* frame = cvQueryFrame(capture); //Create image frames from capture
		cvShowImage("Camera_Output", frame);   //Show image frames on created window

		char key = cvWaitKey(10);     //Capture Keyboard stroke
		if (key == 27){
			break;      //If you hit ESC key loop will break.
		}
	}
	cvReleaseCapture(&capture); //Release capture.
	cvDestroyWindow("Camera_Output"); //Destroy Window
	return 0;
}

int experimental(){
	Mat source_gray;
	Mat dest, edges;
	int counter = 0;

	cvNamedWindow("Camera_Output", 1);    //Create window
	CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);  //Capture using any camera connected to your system
	while (1){ //Create infinte loop for live streaming
		counter++;
		IplImage* frame = cvQueryFrame(capture); //Create image frames from capture

		//cvShowImage("Camera_Output", frame);   //Show image frames on created window

		Mat source(frame);

		if (!source.data)
		{
			return -1;
		}

		dst.create(source.size(), source.type());

		/// Convert the image to grayscale
		cvtColor(source, source_gray, CV_BGR2GRAY);

		/// Create a window
		namedWindow(window_name, CV_WINDOW_AUTOSIZE);

		/// Create a Trackbar for user to enter threshold
		createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, NULL);



		/// Show the image...

		/// Reduce noise with a kernel 3x3
		blur(source_gray, edges, Size(3, 3));

		/// Canny detector
		Canny(edges, edges, lowThreshold, lowThreshold*ratio, kernel_size);

		/// Using Canny's output as a mask, we display our result
		dest = Scalar::all(0);

		source.copyTo(dest, edges);
		if (counter % 20 == 0){
			imwrite("../Tesseract-OCR/test.jpg", dest);
			system("cd ../Tesseract-OCR && tesseract.exe test.jpg output");
			cout << system("cd ../Tesseract-OCR && more output.txt");
		}

		imshow(window_name, dest);



		char key = cvWaitKey(10);     //Capture Keyboard stroke
		if (key == 27){
			break;      //If you hit ESC key loop will break.
		}
	}
	cvReleaseCapture(&capture); //Release capture.
	cvDestroyWindow("Camera_Output"); //Destroy Window
	return 0;
}


int main(int argc, char** argv)
{
	
	/*
	literally this is the worst solution, but this is creating a command prompt that goes up a level, runs tesseract.exe
	on "test.jpg" to "output.txt," then retrieves the contents of output.txt. Requires Tesseract-OCR to be in the correct
	location, and is NOT fast at all.

	TODO: test this on realtime video!

	*/
	//cout << system("cd ../Tesseract-OCR && tesseract.exe test.jpg output && more output.txt");
	//cin.ignore();
	return experimental();
	
}
