#pragma once

#include "stdafx.h"


namespace HandGR{
class NetworkProxy{

	struct timezone{
		int  tz_minuteswest; /* minutes W of Greenwich */
		int  tz_dsttime;     /* type of dst correction */
	};

	struct IPv4{	
		unsigned char b1, b2, b3, b4;
	};

	struct NewClientParams{
		SOCKET clientSocket;
		struct sockaddr_in cliendAddr;
	};

	SOCKET server;

	MouseHandling mouseHandling;

public:
	NetworkProxy();
	void connectToServer( struct sockaddr_in& server_addr );

private:

	bool initWinSocket();

	void move( char * buf, int &numbytes, const unsigned MAXBUFLEN );

	bool createConnection( struct sockaddr_in &server_addr );



};
}