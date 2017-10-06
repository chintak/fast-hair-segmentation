function [Vec] = funcFaceKeypoint(i1,i2,lmk)
lmk = vertcat(lmk, [i2 i1]);
[c,id] = sort(lmk(:,2));
Vec = lmk(id, :);
Vec = double(Vec);
end