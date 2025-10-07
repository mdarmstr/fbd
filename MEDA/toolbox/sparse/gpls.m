function model = gpls(xcs,ycs,states,varargin)

% Group-wise Partial Least Squares. The original paper is Camacho, J., 
% Saccenti, E. Group-wise Partial Least Squares Regression. Submitted to
% Chemometrics and Intelligent Laboratory Systems, 2016.
%
% beta = gpls(xcs,ycs,states)     % minimum call
%
%
% INPUTS:
%
% xcs: [NxM] preprocessed billinear data set 
%
% ycs: [NxO] preprocessed billinear data set of predicted variables
%
% states: {Sx1} Cell with the groups of variables.
%
%
% Optional INPUTS (parameters):
%
% 'LVs': [1xA] Latent Variables considered (e.g. lvs = 1:2 selects the
%   first two LVs). By default, lvs = 0:rank(xcs)
%
% 'Tolerance': [1x1] tolerance value
%
%
% OUTPUT:
%
% model: structure that contains model information
%   var: [1x1] xcs sum of squares
%   lvs: [1xA] latent variable numbers
%   loads: [MxA] matrix of x-loadings P
%   yloads: [OxA] matrix of y-loadings Q
%   weights: [MxA] matrix of weights W
%   altweights: [MxA] matrix of alternative weights R
%   scores: [NxA] matrix of x-scores T
%   beta: [MxO] matrix of regressors
%   type: 'gPLS'
%
%
% EXAMPLE OF USE: Random data:
%
% obs = 20;
% vars = 100;
% X = simuleMV(obs,vars,'LevelCorr',5);
% X = [0.1*randn(obs,5)+X(:,1)*ones(1,5) X(:,6:end)];
% Y = sum((X(:,1:5)),2);
% Y = 0.1*randn(obs,1)*std(Y) + Y;
% lvs = 1;
% map = medaPls(X,Y,'LVs',lvs);
% 
% Xcs = preprocess2D(X,'Preprocessing',2);
% Ycs = preprocess2D(Y,'preprocessing',2);
% [bel,states] = gia(map,'Gamma',0.4,'MinSize',1);
% model = gpls(Xcs,Ycs,states,'LVs',lvs);
% 
% plotVec(model.beta,'XYLabel',{'','Regression coefficients'});
%
% Coded by: Jose Camacho (josecamacho@ugr.es)
% last modification: 1/Jul/2025
% Dependencies: Matlab R2017b, MEDA v1.9
%
% Copyright (C) 2025  University of Granada, Granada
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

%% Arguments checking

% Set default values
routine=dbstack;
assert (nargin >= 3, 'Error in the number of arguments. Type ''help %s'' for more info.', routine.name);
N = size(xcs, 1);
M = size(xcs, 2);
O = size(ycs, 2);

% Introduce optional inputs as parameters (name-value pair) 
p = inputParser;
LVS = 0:rank(xcs);
Tol = 1e-15;
addParameter(p,'LVs',LVS);  
addParameter(p,'Tolerance',Tol);          
parse(p,varargin{:});

% Extract inputs from inputParser for code legibility
lvs = p.Results.LVs;
tol = p.Results.Tolerance;

% Convert column arrays to row arrays
if size(lvs,2) == 1, lvs = lvs'; end;

% Preprocessing
lvs = unique(lvs);
lvs(find(lvs==0)) = [];
lvs(find(lvs>M)) = [];
A = length(lvs);

% Validate dimensions of input data
assert (isequal(size(ycs), [N O]), 'Dimension Error: parameter ''ycs'' must be N-by-O. Type ''help %s'' for more info.', routine.name);
assert (isequal(size(lvs), [1 A]), 'Dimension Error: parameter ''LVs'' must be 1-by-A. Type ''help %s'' for more info.', routine.name);

% Validate values of input data
assert (iscell(states), 'Value Error: parameter ''states'' must be a cell of positive integers. Type ''help %s'' for more info.', routine.name);
for i=1:length(states)
    assert (isempty(find(states{i}<1)) && isequal(fix(states{i}), states{i}), 'Value Error: 3rd argument must be a cell of positive integers. Type ''help %s'' for more info.', routine.name);
    assert (isempty(find(states{i}>M)), 'Value Error: parameter ''states'' must contain values not higher than M. Type ''help %s'' for more info.', routine.name);
end
assert (isempty(find(lvs<0)) && isequal(fix(lvs), lvs), 'Value Error: parameter ''LVs'' must contain positive integers. Type ''help %s'' for more info.', routine.name);



%% Main code

map = xcs'*xcs;
mapy = xcs'*ycs;
I =  eye(M);
B = I;
beta = zeros(M,O);
W = zeros(M,max(lvs));
P = zeros(M,max(lvs));
Q = zeros(O,max(lvs));
T = zeros(N,max(lvs));
bel = zeros(1,max(lvs));
R = zeros(M,max(lvs));
ind = 1;
    
for j = 1:max(lvs)  

    Rt = zeros(M,length(states));
    Tt = zeros(N,length(states));
    Wt = zeros(M,length(states));
    Pt = zeros(M,length(states));
    Qt = zeros(O,length(states));

    for i=1:length(states) % construct eigenvectors according to states
        mapyaux = zeros(size(mapy));
        mapyaux(states{i},:)= mapy(states{i},:);
        if find(mapyaux>tol)
            Wi = zeros(M,1);
            if O == 1
                Wi = mapyaux;
            else
                [C,D] = eig(mapyaux'*mapyaux);
                dd = diag(D);
                if find(dd)
                    Wi = (mapyaux*C(:,find(dd==max(dd))));
                end
            end

            Wt(:,i) = Wi/sqrt(Wi'*Wi);
            Rt(:,i) = B*Wt(:,i); % Dayal & MacGregor eq. (22)
            Tt(:,i) = xcs*Rt(:,i);
        end
    end

    sS = sum((preprocess2D(Tt,'Preprocessing',2)'*ycs).^2,2); % select pseudo-eigenvector with the highest covariance
    if max(sS)
        ind = find(sS==max(sS),1);
    else
        break;
    end
    R(:,j) = Rt(:,ind);
    T(:,j) = Tt(:,ind);
    W(:,j) = Wt(:,ind);
    Q(:,j) = Rt(:,ind)'*mapy/(Tt(:,ind)'*Tt(:,ind));
    P(:,j) = Tt(:,ind)'*xcs/(Tt(:,ind)'*Tt(:,ind));
    bel(j) = ind;
    
	mapy = mapy - P(:,j)*Q(:,j)'*(T(:,j)'*T(:,j));

    q = W(:,j)*P(:,j)';
    B = B*(I-q); 
    
end

% Postprocessing
W = W(:,lvs);
P = P(:,lvs);
Q = Q(:,lvs);
T = T(:,lvs);
bel = bel(lvs);
R = R(:,lvs);
beta=R*Q';

model.var = sum(sum(xcs.^2));
model.lvs = lvs;
model.loads = P;
model.yloads = Q;
model.weights = W;
model.altweights = R;
model.scores = T;
model.beta = beta;
model.type = 'gPLS';