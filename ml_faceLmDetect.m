function lmPts = ml_faceLmDetect(imPath, shldDisp)
% Use DLIB for facial landmark detection
% imPath: path to the image file
% lmPts: 136*nFace
%   lmPts(:,i) is [x1, y1, x2, y2, ...];
% examples:
%   lmPts = ml_faceLmDetect('/Volumes/StorEDGE/Pictures/2016/2016_09_BuoiDauDiHoc/IMG_2807.JPG', 1);
%   ml_faceLmDetect('/Users/hoai/Sites/SBUWeb/pub/photos/cvlab_Feb16b.jpg', 1);
% By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
% Created: 11-Sep-2016
% Last modified: 11-Sep-2016

dlibDir = '/home/csheth/code/hair/dlib/examples';
outFile = sprintf('dlibRlst_%s.txt', ml_randStr());
outFile2 = sprintf('dlibRlst_%s.txt', ml_randStr());

cmd = sprintf('%s/build/m_face_landmark_detection_ex2 %s/shape_predictor_68_face_landmarks.dat %s %s &> %s', ...
    dlibDir, dlibDir, outFile, imPath, outFile2);

% fprintf('%s\n', cmd);
system(cmd);
lmPts = load(outFile);
% lmPts = lmPts/2;
lmPts = lmPts';
cmd = sprintf('rm %s', outFile); system(cmd);
cmd = sprintf('rm %s', outFile2); system(cmd);

if exist('shldDisp', 'var') && shldDisp
   im = imread(imPath);
   imshow(im, 'InitialMagnification', 'fit'); hold on;
   fprintf('Found %d faces\n', size(lmPts,2));
   for i=1:size(lmPts,2)
       triMesh = delaunayTriangulation([lmPts(1:2:end,i), lmPts(2:2:end,i)]);
       scatter(lmPts(1:2:end,i), lmPts(2:2:end,i), '.c');
%        triplot(triMesh, [], [], 'r');
        triplot(triMesh, 'cyan')

   end;
   title(sprintf('Found %d faces\n', size(lmPts,2)));
end
