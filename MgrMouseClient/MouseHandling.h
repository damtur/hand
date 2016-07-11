#pragma once
#include "stdafx.h"
//#include <cxcore.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace HandGR{
	class MouseHandling{

		/** hand move translational to mouse */
		double mouseSensitivity;

		int state;
		
		cv::Mat toShow;
		cv::Mat r;
		cv::Mat t;
		cv::Mat a;
		cv::Mat open;


		/** if moving on or not */
		bool moving;
	public:

		MouseHandling();

		/** Increase hand move translational to mouse */
		void increaseMouseSensitivity();

		/** Decrease hand move translational to mouse */
		void decreaseMouseSensitivity();

		/** Relative move mouse */
		void move(double x, double y);

		/** Send to system right mouse button click */
		void rightClick();

		/** Send to system middle mouse button click*/
		void middleClick();

		/** Send to system left mouse button press */
		void leftDown();

		/** Unclick pressed mouse button (usefull for left button) */
		void nothingAndLeftUp();
	private:
		/** Unclick pressed mouse button */
		void release();

		/** Make gest by system */
		void makeGest(DWORD gest);
	};

}

