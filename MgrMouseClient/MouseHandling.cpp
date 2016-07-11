#include "stdafx.h"

using namespace std;
namespace HandGR{
	
	MouseHandling::MouseHandling(){
		a = cv::imread("a.jpg");
		if(!a.data){
			cout<<"Error: Could not read file a.jpg"<<endl;
		}

		r = cv::imread("r.jpg");
		if(!a.data){
			cout<<"Error: Could not read file r.jpg"<<endl;
		}

		t = cv::imread("t.jpg");
		if(!a.data){
			cout<<"Error: Could not read file t.jpg"<<endl;
		}

		open = cv::imread("open.jpg");
		if(!a.data){
			cout<<"Error: Could not read file open.jpg"<<endl;
		}

		mouseSensitivity = 6000;	
		state = 0;
		moving = false;
	}


	void MouseHandling::increaseMouseSensitivity(){
		if(mouseSensitivity > 100)
			mouseSensitivity -= 100;
	}

	void MouseHandling::decreaseMouseSensitivity(){
		if(mouseSensitivity < 12000)
			mouseSensitivity += 100;
	}

	void MouseHandling::move(double x, double y){
		cout<<"Move ("<<x<<","<<y<<")"<<endl;

 		if(toShow.data){
			cv::Point to(290-(int)(10*x),105+(int)(10*y));
			cv::line(toShow, cv::Point(290,105), to, cv::Scalar(255,255,255),1);
			cv::circle(toShow, to, 2, cvScalar(255,255,255));
			cv::imshow("Recieved", toShow);
			cv::waitKey(1);
		}

		if(moving){

			INPUT *buffer = new INPUT[3]; //allocate a buffer
			buffer->type = INPUT_MOUSE;
			buffer->mi.dx = static_cast<long>(-x * 65535 / mouseSensitivity);
			buffer->mi.dy = static_cast<long>(y * 65535 / mouseSensitivity);
			buffer->mi.mouseData = 0;
			buffer->mi.dwFlags = ( MOUSEEVENTF_MOVE);
			buffer->mi.time = 0;
			buffer->mi.dwExtraInfo = 0;

			SendInput(1,buffer,sizeof(INPUT));
			delete (buffer); //clean up our messes.
		}
	}

	void MouseHandling::rightClick(){
		if(t.data){
			toShow = t.clone();
			cv::imshow("Recieved", toShow);
			cv::waitKey(1);
		}
		

		cout<<"Right click"<<endl;
		if(state != 3){
			release();
			makeGest(MOUSEEVENTF_RIGHTDOWN);
			makeGest(MOUSEEVENTF_RIGHTUP);
			state = 3;
		}
	}

	void MouseHandling::middleClick(){
		if(a.data){
			toShow = a.clone();
			cv::imshow("Recieved", toShow);
			cv::waitKey(1);
		}
		
		cout<<"Middle click"<<endl;
		//if(state != 2){
		//release();
		//makeGest(MOUSEEVENTF_MIDDLEDOWN);
		//makeGest(MOUSEEVENTF_MIDDLEUP);
		moving = true;
		//}
	}

	void MouseHandling::leftDown(){
		if(r.data){
			toShow = r.clone();
			cv::imshow("Recieved", toShow);
			cv::waitKey(1);
		}

		cout<<"Left down"<<endl;
		if(state != 1){
			release();
			makeGest(MOUSEEVENTF_LEFTDOWN);
			//makeGest(MOUSEEVENTF_LEFTUP);
			//auto a=[](){
			//	std::cout<<"fghjgj"<<std::endl;
			//};
			//a();
			//makeGest(MOUSEEVENTF_LEFTUP);
			state = 1;
		}

	}

	void MouseHandling::nothingAndLeftUp(){
		if(open.data){
			toShow = open.clone();
			cv::imshow("Recieved", toShow);
			cv::waitKey(1);
		}
		
		cout<<"Up"<<endl;
		moving = false;
		if(state != 0){
			release();
			state = 0;
		}
	}


	void MouseHandling::release(){
		if(state == 1){
			makeGest( MOUSEEVENTF_LEFTUP);
		}

		//switch (state){
		//case 1: 
		//	
		//	return;
		//case 2:
		//	//makeGest( MOUSEEVENTF_MIDDLEUP);
		//	return;
		//case 3:
		//	makeGest( MOUSEEVENTF_RIGHTUP);
		//	return;
		//}
	}

	void MouseHandling::makeGest(DWORD gest){
		//PlaySound(TEXT("click.wav"), NULL, SND_FILENAME | SND_ASYNC);


		//INPUT *buffer = new INPUT[3]; //allocate a buffer
		//buffer->type = INPUT_MOUSE;
		//buffer->mi.dx = 0; 
		//buffer->mi.dy = 0;
		//buffer->mi.mouseData = 0;

		//buffer->mi.dwFlags = gest;

		//buffer->mi.time = 0;
		//buffer->mi.dwExtraInfo = 0;

		//SendInput(1,buffer,sizeof(INPUT));
		//delete (buffer); //clean up our messes.

	}

}

