/*#ifndef _TEST_APP  
#define _TEST_APP  

#include "ofMain.h"  

#include "ofxOpenCv.h"  
#include "cv.h"  
#include "cvaux.h"  


class testApp : public ofBaseApp{  

public:  

	void setup();  
	void update();  
	void draw();  

	void keyPressed  (int key);  
	void mouseMoved(int x, int y );  
	void mouseDragged(int x, int y, int button);  
	void mousePressed(int x, int y, int button);  
	void mouseReleased(int x, int y, int button);  
	void resized(int w, int h);  


	//ofVideoGrabber      vidGrabber;  

	//ofxCvColorImage     colorImg;  


	CvBGStatModel* gauss_bgModel;  
	//ofxCvGrayscaleImage     gauss_foregroundImg;  
	//ofxCvColorImage     gauss_backgroundImg;  

	CvBGStatModel* fgd_bgModel;  
	//ofxCvGrayscaleImage     fgd_foregroundImg;  
	//ofxCvColorImage     fgd_backgroundImg;  

};  

#endif  */