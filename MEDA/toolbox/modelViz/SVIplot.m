function [r2,alpha,q2,resCV,alphaCV,betas] = SVIplot(x,varargin)

% Structural and Variance Information plots. The original paper is 
% Chemometrics and Intelligent Laboratory Systems 100, 2010, pp. 48-56. 
%
% r2 = SVIplot(x) % minimum call
%
%
% INPUTS:
%
% x: [NxM] billinear data setunder analysis
%
%
% Optional INPUTS (parameter):
%
% 'PCs': [1xA] Principal Components considered (e.g. pcs = 1:2 selects the
%   first two PCs). By default, pcs = 0:rank(xcs)
%
% 'Vars': (1x1) selected variable for the plot (first variable by default)
%
% 'Groups': [1x1] number of groups in the cross-validation run (7 by default)
%
% 'Preprocessing': [1x1] preprocesing of the data
%       0: no preprocessing
%       1: mean centering
%       2: autoscaling (default)   
%
% 'Beta': bool
%       false: SVIplot without beta terms (by default)
%       true: SVIplot plus beta terms
%
%
% OUTPUTS:
%
% r2: Goodness of fit.
% 
% alpha: alpha parameter according to the reference.
%
% q2: Goodness of prediction.
%
% resCV: residuals by CV
%
% alphaCV: alpha by CV
%
%
% EXAMPLE OF USE: Random data
%
% X = simuleMV(20,10,'LevelCorr',8);
% var = 1;
% [r2,alpha,q2,resCV,alphaCV] = SVIplot(X,'PCs',1:3,'Vars',var);
%
%
% coded by: Jose Camacho (josecamacho@ugr.es)
% last modification: 15/Jan/2025
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

%% Parameters checking

% Set default values
routine=dbstack;
assert (nargin >= 1, 'Error in the number of arguments. Type ''help %s'' for more info.', routine(1).name);
N = size(x, 1);
M = size(x, 2);

% Introduce optional inputs as parameters (name-value pair) 
p = inputParser;
addParameter(p,'PCs',0:min(size(x)));   
addParameter(p,'Vars',1);   
addParameter(p,'Groups',7);
addParameter(p,'Preprocessing',2);
addParameter(p,'Beta',false);
parse(p,varargin{:});

% Extract inputs from inputParser for code legibility
pcs = p.Results.PCs;
var = p.Results.Vars;
groups = p.Results.Groups;
beta = p.Results.Beta;
prep = p.Results.Preprocessing;

% Convert column arrays to row arrays
if size(pcs,2) == 1, pcs = pcs'; end;

% Preprocessing
pcs = unique(pcs);
pcs(find(pcs==0)) = [];
pcs(find(pcs>min(size(x)))) = [];
A = length(pcs);

% Validate dimensions of input data
assert (isequal(size(pcs), [1 A]), 'Dimension Error: parameter ''PCs'' must be 1-by-A. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(var), [1 1]), 'Dimension Error: parameter ''Vars'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(groups), [1 1]), 'Dimension Error: parameter ''Groups'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(prep), [1 1]), 'Dimension Error: parameter ''Preprocessing'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);
  
% Validate values of input data
assert (isempty(find(pcs<0)) && isequal(fix(pcs), pcs), 'Value Error: parameter ''PCs'' must contain positive integers. Type ''help %s'' for more info.', routine(1).name);


%% Main code

xcs = preprocess2D(x,'preprocessing',prep);
model = pcaEig(xcs,'PCs',1:max(pcs));
p = model.loads;

alpha=0;
betas=zeros(M-1,1);
r2 = 0;
for i=1:length(pcs)
    q = p(:,1:pcs(i))*p(:,1:pcs(i))';
    alpha = [alpha q(var,var)];
    betas = [betas q([1:var-1 var+1:end],var)];
    res = xcs*(eye(M)-q);
    r2 = [r2 1-sum(res(:,var).^2)/sum(xcs(:,var).^2)];
end

resCV=[];
alphaCV=[];
pcsvect=[];
rows = rand(1,N);
[a,rind]=sort(rows);
elemr=N/groups;
for j=1:groups
    indi = rind(round((j-1)*elemr+1):round(j*elemr)); % Sample selection
    i2 = ones(N,1);
    i2(indi)=0;
    test = x(indi,:);
    cal = x(find(i2),:);
    st = size(test);

    [calc,m,sd] = preprocess2D(cal,'Preprocessing',prep);
    testc = preprocess2Dapp(test,m,'Scale',sd);
    
    model = pcaEig(calc,'PCs',1:max(pcs));
    p = model.loads;
    alpha2=0;
    res2 = testc(:,var);
    for i=1:length(pcs)
        kk = p(:,1:min(pcs(i),size(p,2)));
        q = kk*kk';
        alpha2 = [alpha2 q(var,var)];
        res = testc*(eye(M)-q);
        res2 = [res2 res(:,var)];
    end
    
    alphaCV=[alphaCV;alpha2];
    pcsvect=[pcsvect;[0 pcs]];
    resCV=[resCV;res2];
    
end

q2 = 1-sum(resCV.^2)/sum(resCV(:,1).^2);


%% Show results

figh=figure;
hold on
plot([0 pcs],r2,'.-');
plot([0 pcs],alpha,'g-x','LineWidth',2);
plot([0 pcs],q2,'m.-');
plot(pcsvect(1:end),alphaCV(1:end),'ro');
chh=get(figh,'Children');
set(chh,'FontSize',14)
legend('R^2_{A,m}','\alpha^A_{m}','Q^2_{A,m}','\alpha^A_{m}(i)','Location','NorthOutside','Orientation','Horizontal')
if beta
    plot([0 pcs],betas','c')
end

% Set axis
axis tight
ax = axis;
axis auto
ax2 = axis;
axis([ax(1:2) ax2(3:4)])

%legend off
box on
hold off


