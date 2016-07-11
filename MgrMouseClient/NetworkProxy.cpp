#include "stdafx.h"

namespace HandGR{
	NetworkProxy::NetworkProxy(){
		if(!initWinSocket()){
			//Error
			std::cerr<<"error in initializing WINsockets"<<std::endl;//TODO
			return;
		}
	}

	bool NetworkProxy::initWinSocket(){
		WSADATA wsaData;
		int error;
	
		error = WSAStartup(  MAKEWORD( 2, 2 ), &wsaData );

		/* check for error */
		if ( error != 0 )
		{
			/* error occured */
			return false;
		}

		/* check for correct version */
		if ( LOBYTE( wsaData.wVersion ) != 2 ||
			HIBYTE( wsaData.wVersion ) != 2 )
		{
			/* incorrect WinSock version */
			WSACleanup();
			return false;
		}
		return true;
		/* WinSock has been initialized */
	}

	void NetworkProxy::connectToServer( struct sockaddr_in& server_addr ){
		const unsigned MAXBUFLEN = 10000;
		char buf[MAXBUFLEN];

		int numbytes;  

		if(!createConnection(server_addr)){
			std::cout<<"No server found."<<std::endl;
			return;
		}

		fd_set master;
		struct timeval tv;

		FD_ZERO(&master);  
		FD_SET(server, &master);

		//wait for responses
		//CLIENT LOOP
		while(true){
			tv.tv_sec = 1;//Set timeout to 1 second
			tv.tv_usec = 0;
			FD_SET(server, &master);

			//Check if serwer is still there
			if (select(server+1, &master, NULL, NULL, &tv) == SOCKET_ERROR) {
				closesocket(server);
				server = NULL;
				std::cout<<"Server connection lost trying to establish new connection"<<std::endl;
				return;
			}

			//Receive information from server
			if( FD_ISSET(server, &master) ){
				if ((numbytes=recv(server, buf, MAXBUFLEN-1, 0)) == -1) {
					//tu wchodze jak server padnie
					closesocket(server);
					server = NULL;
					std::cout<<"Server connection lost trying to establish new connection"<<std::endl;
					return;
				}
				if(numbytes == 0){
					closesocket(server);
					std::cout<<"Server connection lost trying to establish new connection"<<std::endl;
					return;
				}
				switch(buf[0]){
				case DATAGRAM_MOVE:{
					move(buf, numbytes, MAXBUFLEN);
					break;
								   }

				case DATAGRAM_R:{
					mouseHandling.leftDown();
					break;
								}

				case DATAGRAM_A:{
					mouseHandling.middleClick();
					break;
								}

				case DATAGRAM_T:{
					mouseHandling.rightClick();
					break;
								}

				case DATAGRAM_NOTHING:{
					mouseHandling.nothingAndLeftUp();
					break;
									  }

				case DATAGRAM_INCREASE_SENSITIVITY:{
					mouseHandling.increaseMouseSensitivity();
					break;
												   }

				case DATAGRAM_DECREASE_SENSITIVITY:{
					mouseHandling.decreaseMouseSensitivity();
					break;
												   }
				}
			}else{
				//std::cout<<"Client timeout...!"<<std::endl;
			}
		}
	}

	bool NetworkProxy::createConnection( struct sockaddr_in &server_addr ){
		std::cout<<"Trying to connect..."<<std::endl;

		server_addr.sin_port = htons( SERVER_PORT );

		if ((server = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
			perror("socket");
			exit(1);//Nieutworzenie soketa jest bardzo ma³o prawdopodobne ale krytyczne
		}

		if (connect(server, (struct sockaddr *)&server_addr, sizeof(struct sockaddr)) == -1) {
			//perror("Connect to server error");
			closesocket(server);
			server = NULL;
			//Jednak nie ma serwera 
			return false;//Bedziemy go Tworzyc jeszcze raz
		}
		return true;
	}

	void NetworkProxy::move( char * buf, int &numbytes, const unsigned MAXBUFLEN ){
		int tempPacketSize = sizeof(char) + (2 * sizeof(double));
		while(numbytes < tempPacketSize){
			int tempBytes;
			if ((tempBytes=recv(server, buf+numbytes, MAXBUFLEN-1, 0)) == -1) {
				std::cout<<"Server connection lost trying to establish new connection"<<std::endl;
				return;
			}
			if(tempBytes == 0){
				closesocket(server);
				std::cout<<"Server connection lost trying to establish new connection"<<std::endl;
				return;
			}
			numbytes+=tempBytes;
		}

		double dx;
		memcpy(&dx , buf+1, sizeof(double));

		double dy;
		memcpy(&dy , buf + sizeof(double) + 1, sizeof(double));

		mouseHandling.move(dx, dy);
	}

}