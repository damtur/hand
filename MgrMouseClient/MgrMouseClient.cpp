// MgrMouseClient.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace HandGR;

int main(){
	NetworkProxy networkProxy;

	struct sockaddr_in sin;

	memset( &sin, 0, sizeof sin );

	sin.sin_family = AF_INET;
	sin.sin_addr.s_addr = inet_addr("127.0.0.1");//INADDR_ANY;

	while(true){
		networkProxy.connectToServer(sin);
		std::cout<<"Could not establish connection. Trying again... (in 5sec)"<<std::endl;
		Sleep(DWORD(5000));
	}

	return 0;
}

