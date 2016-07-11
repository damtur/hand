namespace HandGR{
	class SimpleSkinDetectionFunction: public SkinDetectionFunction{
	public:
		bool isSkin(uchar r, uchar g, uchar b){
			return true;
		}

	};
}