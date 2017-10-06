[cDat,dDat] = readtext('LFW/TrainDatNames1.txt');
[cImg,dImg] = readtext('LFW/TrainImgNames1.txt');
[cGT,dGT] = readtext('LFW/TrainDatNames1.txt');

% GT Labels hair = 0; Face =1; Background = 2;

NoImages = size(cImg,1);
Side = 11;
nameKey = strcat('LFW/FaceKeypoints_',num2str(Side),'_',num2str(NoImages),'_Train.mat');
upsize = 500;
%%%%%%%%%%%%%%%%%%%%%%%%%%%
Keypoints = cell(NoImages,1);
Names = cell(NoImages,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
cnt250=0;
parfor i = 1:NoImages
	namex = cGT{i};
	fprintf('%d = %s ', i, cImg{i});
	[xx1,xx2,xx3] = fileparts(namex);
	xx5 = strsplit(xx1,'/');

	Img = imread(cImg{i});
        test_name = sprintf('test_fk%d.jpg', i);
	imwrite(Img, test_name);
	lmknew = ml_faceLmDetect(test_name);

	if(size(lmknew,2)>0)
            if(size(lmknew,2)==1) % currently handle single faces per image
		lmk= lmknew(1:2:end);
		lmk = horzcat(lmk,lmknew(2:2:end));
            else
		lmknew =lmknew(:,1);
		lmk= lmknew(1:2:end);
		lmk = horzcat(lmk,lmknew(2:2:end));
            end
            [m,n,c] = size(Img);
            Keypoints{i} = funcFaceKeypoint(m,n,lmk);
            Names{i} = cImg{i};
            fprintf('... done\n');
	end
        delete(test_name);
end
if(numel(Keypoints) > 1)
    save(nameKey,'Keypoints', 'Names');
end
