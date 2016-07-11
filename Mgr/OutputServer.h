#pragma once

#include "stdafx.h"


namespace HandGR{

	class OutputServer{
	public:
		struct ClientData{
			SOCKET clientSocket;
			struct sockaddr_in cliendAddr;
		};
	private:

		static HANDLE hMutex;
		static OutputServer* outpusServer;
		static SOCKET serverSocket;

		struct sockaddr_in serverAddress;

		OutputServer();

		static std::shared_ptr<std::vector<ClientData> > clients;


	public:
		static OutputServer* getInstance();

		void move(double dx, double dy);

		void gestA();
		void gestT();
		void gestR();
		void gestNothing();
		void increaseSensitivity();
		void decreaseSensitivity();

	private:
		void initSockets();
		void createServer();
		void send(const char gest);
		bool send(SOCKET client, const char gest );
		bool move(SOCKET client, double dx, double dy);

		static void serverAcceptingThread( void* pParams );
		static void clientThread( void* pParams);
	};
}