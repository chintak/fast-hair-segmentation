function randStr = ml_randStr()

s = ...
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';

%find number of random characters to choose from
numRands = numel(s);

%specify length of random string to generate
sLength = 7;

%generate random string
randStr = s( ceil(rand(1,sLength)*numRands) );

end