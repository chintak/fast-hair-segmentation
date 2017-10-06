function [Vec] = funcRgbHogPos(Img,i1,i2,Side,lmk)

	[c,id] = sort(lmk(:,2));
	faceheight = c(end)-c(1);
	pos = [lmk(id(1),2) lmk(id(1),1)];
	Pos2 = [pos(1) - i1, pos(2) - i2];
	Pos2 = Pos2/faceheight;
	Patch = Img(i1:i1+Side-1,i2:i2+Side-1,:);
	pat = rgb2gray(Patch);
	hog = vl_hog(single(pat), Side) ;
	hog=reshape(hog,[1,31]);
	Vec = reshape(Patch,[1,Side*Side*3]);
	Vec = double(Vec);
	Vec = horzcat(Pos2,Vec,hog);
	
	
end