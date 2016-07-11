#include "stdafx.h"

/** Simple skin detection function in RGB colour space */
namespace HandGR{
	class SimpleSkinDetectionFunction5: public SkinDetectionFunction{
	public:
		SimpleSkinDetectionFunction5(bool initialFilterEnabled = true) : SkinDetectionFunction(initialFilterEnabled){}


	};
}