t = str2double(split(fileread("t.txt")));
x = str2double(split(fileread("x.txt")));

N = height(t);

H = [ones(N,1),t,sin(2*pi*t)];


Theta_hat = H\x

