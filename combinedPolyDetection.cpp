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
#include "shape-detect.cpp"
#include <stdio.h>
#include <stdlib.h>

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

void polyDetect(Mat image)
{

  Mat img = image;
  
 //show the original image
 //cvNamedWindow("Raw");
 //cvShowImage("Raw",img);

  //converting the original image into grayscale
 Mat imgGrayScale; 
 cvtColor(img,imgGrayScale,CV_BGR2GRAY);

  //thresholding the grayscale image to get better results
 threshold(imgGrayScale,imgGrayScale,128,255,CV_THRESH_BINARY);  
 
 vector<vector<Point> > contours;  //hold the pointer to a contour in the memory block
 vector<Vec4i> hierarchy;   //hold sequence of points of a contour
 //CvMemStorage *storage = cvCreateMemStorage(0); //storage area for all contours
 
 //finding all contours in the image
 findContours(imgGrayScale, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
  
 //iterating through each contour
 for(int i = 0; i< contours.size(); i++)
 {
     //obtain a sequence of points of contour, pointed by the variable 'contour'
	 vector<Point> result;
	 approxPolyDP(contours[i], result, arcLength(Mat(contours[i]), true)*0.02, true);
           
     //if there are 3  vertices  in the contour(It should be a triangle)
    if(result.size()==3 )
     {
         //iterating through each point
         //CvPoint *pt[3];
         //for(int i=0;i<3;i++){
             //pt[i] = (CvPoint*)cvGetSeqElem(result, i);
         //}
   
         //drawing lines around the triangle
         line(img, result.at(0), result.at(1), cvScalar(255,0,0),4);
         line(img, result.at(1), result.at(2), cvScalar(255,0,0),4);
         line(img, result.at(2), result.at(0), cvScalar(255,0,0),4);
       
     }

      //if there are 4 vertices in the contour(It should be a quadrilateral)
     else if(result.size()==4 )
     {
         //iterating through each point
         //CvPoint *pt[4];
         //for(int i=0;i<4;i++){
             //pt[i] = (CvPoint*)cvGetSeqElem(result, i);
         //}

         //drawing lines around the quadrilateral
         line(img, result.at(0), result.at(1), cvScalar(255,0,0),4);
         line(img, result.at(1), result.at(2), cvScalar(255,0,0),4);
         line(img, result.at(2), result.at(3), cvScalar(255,0,0),4);
         line(img, result.at(3), result.at(0), cvScalar(255,0,0),4);
         
     
     }
     
    else if(result.size()==7 )
     {
         //iterating through each point
         //CvPoint *pt[4];
         //for(int i=0;i<4;i++){
             //pt[i] = (CvPoint*)cvGetSeqElem(result, i);
         //}

         //drawing lines around the heptagon
         line(img, result.at(0), result.at(1), cvScalar(255,0,0),4);
         line(img, result.at(1), result.at(2), cvScalar(255,0,0),4);
         line(img, result.at(2), result.at(3), cvScalar(255,0,0),4);
         line(img, result.at(3), result.at(4), cvScalar(255,0,0),4);
         line(img, result.at(4), result.at(5), cvScalar(255,0,0),4);
         line(img, result.at(5), result.at(6), cvScalar(255,0,0),4);
         line(img, result.at(6), result.at(0), cvScalar(255,0,0),4);
         
     
     }

 }

  //show the image in which identified shapes are marked   
 //cvNamedWindow("Tracked");
 //cvShowImage("Tracked",img);
   
 //cvWaitKey(0); //wait for a key press

  //cleaning up
 //cvDestroyAllWindows(); 
 //cvReleaseMemStorage(&storage);
 //cvReleaseImage(&img);
 //cvReleaseImage(&imgGrayScale);

}

void houghImage(Mat img){
	Mat src = img;

	Mat dst, cdst;
	Canny(src, dst, 50, 200, 3); 
	cvtColor(dst, cdst, CV_GRAY2BGR); 

	vector<Vec2f> lines;
	// detect lines
	HoughLines(dst, lines, 1, CV_PI/180, 150, 0, 0 );

	// draw lines
	for( size_t i = 0; i < lines.size(); i++ )
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		line(src, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
	}
}

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

int lineDetect(Mat img){
	Mat source_gray;
	Mat dest, edges;

cvNamedWindow("Camera_Output", 1);    //Create window

	//cvShowImage("Camera_Output", frame);   //Show image frames on created window

	Mat source = img;

	if (!source.data)
	{
		cout << "You suck \n";
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
	
	imwrite("out.jpg", dest);

	imshow(window_name, dest);


	waitKey(0);
	cvDestroyWindow("Camera_Output"); //Destroy Window
	return 0;
}

int stream(){
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
		
		houghImage(dest);

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

int main () {
	return stream();
	return 0;
}
