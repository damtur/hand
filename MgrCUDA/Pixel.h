#pragma once

#ifdef MYAPI_EXPORTS
#define MYAPI_API __declspec(dllexport)
#else
#define MYAPI_API __declspec(dllimport)
#endif


struct MYAPI_API Pixel{
	unsigned char b;
	unsigned char g;
	unsigned char r;
};