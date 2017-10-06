[cDat,dDat] = readtext('LFW/TrainDatNames1.txt');
[cImg,dImg] = readtext('LFW/TrainImgNames1.txt');
[cGT,dGT] = readtext('LFW/TrainDatNames1.txt');

% GT Labels hair = 0; Face =1; Background = 2;

NoImages = size(cImg,1);
Side = 11;
nameHair = strcat('LFW/Patches/Hair_',num2str(Side),'_',num2str(NoImages),'_TrainPatches_noUpsampling_All.mat');
nameFace = strcat('LFW/Patches/Face_',num2str(Side),'_',num2str(NoImages),'_TrainPatches_noUpsampling_All.mat');
nameBackground = strcat('LFW/Patches/Background_',num2str(Side),'_',num2str(NoImages),'_TrainPatches_noUpsampling_All.mat');
upsize = 500;
%%%%%%%%%%%%%%%%%%%%%%%%%%%
Hair = [];
Background = [];
Face = [];
HairXY = [];
BackgroundXY = [];
FaceXY = [];
Haircnter = 0;
Backgroundcnter = 0;
Facecnter = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
cnt250=0;
for i = 1:NoImages
	namex = cGT{i};
	fprintf('%d = %s ', i, namex);
	[xx1,xx2,xx3] = fileparts(namex);
	xx5 = strsplit(xx1,'/');
	namexx = strcat('LFW/gt/parts_lfw_funneled_superpixels_mat/',xx5(end),'/',xx2,xx3);
	SupPixel = load(char(namexx));
	%SupPixel = floor(impyramid(SupPixel, 'expand'));

	%SupPixel = imresize(SupPixel,[500,500]);
	Img = imread(cImg{i});
	%Img = imresize(Img,[upsize,upsize]);
	%Img = impyramid(Img, 'expand');
	imwrite(Img,'test39.jpg');
	lmknew = ml_faceLmDetect('test39.jpg');
	if(size(lmknew,2)>0)
	if(size(lmknew,2)==1) % currently handle single faces per image
		lmk= lmknew(1:2:end);
		lmk = horzcat(lmk,lmknew(2:2:end));
	else
		lmknew =lmknew(:,1);
		lmk= lmknew(1:2:end);
		lmk = horzcat(lmk,lmknew(2:2:end));
	end


	SupPixelLabel = load(cDat{i});
	noSupPixels = SupPixelLabel(1);
	SupPixel1 = SupPixel;
	for j = 1:noSupPixels
		label = SupPixelLabel(j+1);
		splabel = j-1;
		SupPixel1(SupPixel==splabel)=label;
	end
	for i1 = 1:Side:250
		for i2 = 1:Side:250
			if(i1+Side-1<=250 && i2+Side-1<=250)
				Vec = funcRgbHogPos(Img,i1,i2,Side,lmk);
				PatchLabels = SupPixel1(i1:i1+Side-1,i2:i2+Side-1);
				if(nnz(PatchLabels==2)/(Side*Side)>0.8) % patch is considered background if 80% pixels are background
					if(1)
					Backgroundcnter = Backgroundcnter + 1;
					Background(Backgroundcnter,:) = Vec;
					BackgroundXY(Backgroundcnter,:) = [i1,i2,i];
					end

				elseif(nnz(PatchLabels==0)/(Side*Side)>0.8)
					if(1)
					Haircnter = Haircnter + 1;
					Hair(Haircnter,:) = Vec;
					HairXY(Haircnter,:) = [i1,i2,i];
					end

				elseif(nnz(PatchLabels==1)/(Side*Side)>0.8)
					if(1)
					Facecnter = Facecnter + 1;
					Face(Facecnter,:) = Vec;
					FaceXY(Facecnter,:) = [i1,i2,i];
					end
				end
			end
		end
	end
	fprintf('... done\n', i, namex);
	if(i==100)
		save(nameFace,'Face','FaceXY','Facecnter','-v7.3');
		save(nameHair,'Hair','HairXY','Haircnter','-v7.3');
		save(nameBackground,'Background','BackgroundXY','Backgroundcnter','-v7.3');

	end
	end
end
save(nameFace,'Face','FaceXY','Facecnter','-v7.3');
save(nameHair,'Hair','HairXY','Haircnter','-v7.3');
save(nameBackground,'Background','BackgroundXY','Backgroundcnter','-v7.3');
