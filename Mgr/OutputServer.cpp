#include "stdafx.h"

#pragma warning(disable: 4127)

namespace HandGR{

	HANDLE OutputServer::hMutex;
	OutputServer* OutputServer::outpusServer = NULL;
	SOCKET OutputServer::serverSocket;

	std::shared_ptr<std::vector<OutputServer::ClientData> > OutputServer::clients;

	OutputServer* OutputServer::getInstance(){
		if (outpusServer == NULL){
			hMutex = CreateMutex( NULL, FALSE, NULL );
			outpusServer = new OutputServer();
		}
		return outpusServer;
	}

	OutputServer::OutputServer(){
		clients = shared_ptr<vector<ClientData> >(new vector<ClientData>());
		initSockets();
		createServer();

	}

	void OutputServer::initSockets(){
		WSADATA wsaData;

		int starterr = WSAStartup(MAKEWORD(2,2), &wsaData); // requests Winsocket version 2.2
		if (starterr != 0) {
			cout << "Error: " << WSAGetLastError() << " occurred!" << endl;
			WSACleanup();
			return;
		}
		cout << "WSAStartup Success!" << endl;
	}
	void OutputServer::createServer(){
		std::cout<<"Starting output server on current machine."<<std::endl;

		serverSocket = socket( AF_INET, SOCK_STREAM, 0 );

		//struct sockaddr_in sin;
		//memset( &sin, 0, sizeof sin );

		serverAddress.sin_family = AF_INET;     // host byte order
		serverAddress.sin_port = htons(SERVER_PORT); // short, network byte order
		serverAddress.sin_addr.s_addr = INADDR_ANY;//inet_addr("127.0.0.1");//INADDR_BROADCAST;
		memset(&(serverAddress.sin_zero), '\0', 8); // wyzeruj resztê struktury



		//sin.sin_family = AF_INET;
		//sin.sin_addr.s_addr = INADDR_ANY;
		//sin.sin_port = htons( SERVER_PORT );

		if ( bind( serverSocket, reinterpret_cast<const sockaddr*>(&serverAddress), sizeof serverAddress ) == SOCKET_ERROR ){
			//Error
			std::cerr<<"Could not start server " <<std::endl;//TODO
			return;
		}

		if (listen(serverSocket, SOMAXCONN ) == -1) {
			perror("Listen server error");
			return;
		}

		_beginthread(serverAcceptingThread, 0, NULL);

	}



	void OutputServer::serverAcceptingThread( void* pParams ){
		//This thread is server side thread which listens is thera are any new clients
		fd_set master;
		struct timeval tv;

		FD_ZERO(&master);  
		FD_SET(serverSocket, &master);

		//wait for responses
		while(true){
			tv.tv_sec = TIMEOUT_COMMON;
			tv.tv_usec = 0;
			FD_SET(serverSocket, &master);

			if (select(serverSocket+1, &master, NULL, NULL, &tv) == SOCKET_ERROR) {
				perror("select in server accepting thread");
			}

			if( FD_ISSET(serverSocket, &master) ){
				struct sockaddr_in their_addr; // informacja o adresie osoby ³¹cz¹cej siê
				int sin_size;
				SOCKET clientSocket;

				sin_size = sizeof(struct sockaddr_in);
				if ((clientSocket = accept(serverSocket, (struct sockaddr *)&their_addr, &sin_size)) == -1) {
					perror("accept");
					continue;
				}
				printf("server: got connection from %s\n", inet_ntoa(their_addr.sin_addr));

				ClientData params;
				params.clientSocket = clientSocket;
				params.cliendAddr = their_addr;

				clients->push_back(params);
				//_beginthread(clientThread, 0, &params);

			}else{
				//std::cout<<"Timeout in server accepting thread (checking if there are only one server)"<<std::endl;
				//	closesocket(serverSocket);
				//	serverSocket = NULL;
				//	killThisServer = false;
				//	_endthread();
				//	return;
			}
		}
		_endthread();
	}

	bool OutputServer::send( SOCKET client, const char gest ){
		int numbytes;
		char datagram[sizeof(char)];
		memcpy(datagram, &gest, sizeof(char));

		if ((numbytes=sendto(client, datagram, sizeof(char), 0,
			(struct sockaddr *)&serverAddress, sizeof(struct sockaddr))) == -1) {
				perror("sendto w Present diagram");
				return false;
		}
		if(numbytes != sizeof(char)){
			perror("sento-not everything sent");
			return false;
		}
		return true;
	}

	//////////////////////////////////////////////////////////////////////////
	//DATAGRAMS
	//////////////////////////////////////////////////////////////////////////







	bool OutputServer::move(SOCKET client, double dx, double dy){
		int numbytes;

		char datagram[sizeof(char) + (sizeof(double) * 2)];
		char moveDescriptor = DATAGRAM_MOVE;
		memcpy(datagram, &moveDescriptor, sizeof(char));
		memcpy(datagram+sizeof(char), &dx, sizeof(double));
		memcpy(datagram+sizeof(char)+sizeof(double), &dy , sizeof(double));

		if ((numbytes=sendto(client, datagram, sizeof(char) + sizeof(double) * 2, 0,
			(struct sockaddr *)&serverAddress, sizeof(struct sockaddr))) == -1) {
				perror("sendto w Present diagram");
				return false;
		}
		if(numbytes != (sizeof(char) + sizeof(double) * 2)){
			perror("sento-not everything sent");
		}
		return true;
	}


	//////////////////////////////////////////////////////////////////////////
	// PUBLIC
	//////////////////////////////////////////////////////////////////////////
	void OutputServer::send(const char gest){
		if(!clients->empty()){
			vector<ClientData>::iterator it;
			for( it=clients->begin(); it!=clients->end(); ){
				if(send(it->clientSocket, gest)){
					++it;
				}else{
					it = clients->erase(it);
				}
			}
		}else{
			cout<<"No clients to send to"<<endl;
		}

	}



	void OutputServer::move( double dx, double dy ){
		if(!clients->empty()){
			vector<ClientData>::iterator it;
			for( it=clients->begin(); it!=clients->end();  ){
				if(move(it->clientSocket, dx, dy)){
					++it;
				}else{
					it = clients->erase(it);
				}
			}
		}else{
			cout<<"No clients to send to"<<endl;
		}
	}


	void OutputServer::gestA(){
		send(DATAGRAM_A);
	}

	void OutputServer::decreaseSensitivity(){
		send(DATAGRAM_DECREASE_SENSITIVITY);
	}

	void OutputServer::increaseSensitivity(){
		send(DATAGRAM_INCREASE_SENSITIVITY);
	}

	void OutputServer::gestNothing(){
		send(DATAGRAM_NOTHING);
	}

	void OutputServer::gestR(){
		send(DATAGRAM_R);
	}

	void OutputServer::gestT(){
		send(DATAGRAM_T);
	}


}
