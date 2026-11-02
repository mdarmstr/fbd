function rv = rv_coefficient(X,Y)
X = X - mean(X,1);
Y = Y - mean(Y,1);
XY = X'*Y;
rv = trace(XY*XY') / sqrt(trace((X'*X)^2)*trace((Y'*Y)^2));
end