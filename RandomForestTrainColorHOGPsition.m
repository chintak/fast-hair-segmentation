
NoImages = 1500;
Side = 11;
addpath(genpath('/Piotr/toolbox'))
nameHair = strcat('/LFW/Patches/Hair_',num2str(Side),'_',num2str(NoImages),'_TrainPatches_noUpsampling_All.mat');

nameFace = strcat('/LFW/Patches/Face_',num2str(Side),'_',num2str(NoImages),'_TrainPatches_noUpsampling_All.mat');
nameBackground = strcat('/LFW/Patches/Background_',num2str(Side),'_',num2str(NoImages),'_TrainPatches_noUpsampling_All.mat');
upsize = 500;
upsize = 500;



load(nameFace)
load(nameHair)
load(nameBackground)

BackgroundTrain = Background;
%BackgroundXYTrain = BackgroundXY(1:50000,:);

%BackgroundRem = Background(50001:end,:);
%BackgroundXYRem = BackgroundXY(50001:end,:);


Xtrain = [];
Ytrain = [];
Xvalid = [];
Yvalid = [];

X = Hair;
Y = 1*ones(size(X,1),1);
RandPosId = (randperm(size(Y,1)))';
id=floor(0.8*size(Y,1));
TrainId = (RandPosId(1:id));
ValidId = (RandPosId(id+1:end));
Xtrain = vertcat(Xtrain,X(TrainId,:));
Ytrain = vertcat(Ytrain,Y(TrainId));
Xvalid = vertcat(Xvalid,X(ValidId,:));
Yvalid = vertcat(Yvalid,Y(ValidId));


X = BackgroundTrain;
Y = 2*ones(size(X,1),1);
RandPosId = (randperm(size(Y,1)))';
id=floor(0.8*size(Y,1));
TrainId = (RandPosId(1:id));
ValidId = (RandPosId(id+1:end));
Xtrain = vertcat(Xtrain,X(TrainId,:));
Ytrain = vertcat(Ytrain,Y(TrainId));
Xvalid = vertcat(Xvalid,X(ValidId,:));
Yvalid = vertcat(Yvalid,Y(ValidId));

X = Face;
Y = 3*ones(size(X,1),1);
RandPosId = (randperm(size(Y,1)))';
id=floor(0.8*size(Y,1));
TrainId = (RandPosId(1:id));
ValidId = (RandPosId(id+1:end));
Xtrain = vertcat(Xtrain,X(TrainId,:));
Ytrain = vertcat(Ytrain,Y(TrainId));
Xvalid = vertcat(Xvalid,X(ValidId,:));
Yvalid = vertcat(Yvalid,Y(ValidId));

Xtrain(:,3:3+(Side*Side*3)-1)=Xtrain(:,3:3+(Side*Side*3)-1)/255;

Xvalid(:,3:3+(Side*Side*3)-1)=Xvalid(:,3:3+(Side*Side*3)-1)/255;

%------------Random Forest------------------------
out=[];
for i = 160:20:220
 pTrain={'maxDepth',30,'F1',i,'M',20,'minChild',1,'N1',80000};
 forest=forestTrain(single(Xtrain),Ytrain,pTrain{:});
 pred = forestApply(single(Xvalid),forest);
 acc = 100*(nnz(pred==Yvalid)/size(Yvalid,1))
 temp = [i,acc];
 out=vertcat(out,temp)
 end






%namemodel ='/LFW/Forest_Color_HOG_position_no_upsampling_500K.mat'
namemodel ='/LFW/Forest_Color_HOG_position_no_upsampling_500K_F1_180_M15_N100000.mat'

X = vertcat(Xtrain,Xvalid);
 Y = vertcat(Ytrain,Yvalid);
  pTrain={'maxDepth',30,'F1',180,'M',15,'minChild',1,'N1',100000};
  forest=forestTrain(single(X),Y,pTrain{:});
   %save(namemodel,'forest','pTrain')




