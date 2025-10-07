
function [Dstt,Qstt,Rt,Rq] = mspcAdicov(Lmodel,varargin)

% Compute D-st and Q-st in Covariance MSPC using ADICOV. Unlike e.g. mspcLpca, 
% here we do not compute the statistics of calibration data or control limits,
% since a set of Lmodels would be needed for that.
%
% mspcAdicov(Lmodel,test) % minimum call
%
%
% INPUTS:
%
% Lmodel: (struct Lmodel) model with the information to compute the PCA
%   model:
%       Lmodel.XX: [MxM] X-block cross-product matrix.
%       Lmodel.lvs: [1x1] number of PCs. 
%       Lmodel.av: [1xM] sample average according to the preprocessing method.
%       Lmodel.sc: [1xM] sample scale according to the preprocessing method.
%       Lmodel.weight: [1xM] weight applied after the preprocessing method.
%
% Optional INPUTS (parameters):
%
% 'Test': [LxM] data set with the observations to be compared. These data 
%   are preprocessed in the same way than calibration data
% 
% 'Index': (1x1) MSPC index definition
%       0: ADICOV similarity index according to Chemometrics and Intelligent 
%           Laboratory Systems 105, 2011, pp. 171-180 (default)
%       1: Modified index 
%
%
% OUTPUTS:
%
% Dstt: [1x1] D-statistic or Hotelling T2 of test
%
% Qstt: [1x1] Q-statistic of test
%
% Rt: [LxM] Differential matrix for diagnosing the D statistics 
%
% Rq: [LxM] Differential matrix for diagnosing the Q statistics  
%
%
% EXAMPLE OF USE: ADICOV-based MSPC on NOC test data and anomalies.
%
% nobs = 100;
% nvars = 10;
% nPCs = 1;
% Lmodel = iniLmodel(simuleMV(nobs,nvars,'LevelCorr',6));
% Lmodel.multr = 100*rand(nobs,1); 
% Lmodel.lvs = 1:nPCs;
% 
% nobst = 10;
% test = simuleMV(nobst,nvars,'LevelCorr',6,'Covar',corr(Lmodel.centr)*(nobst-1)/(Lmodel.N-1));
% test(6:10,:) = 3*test(6:10,:);
% 
% [Dstt,Qstt] = mspcAdicov(Lmodel,'Test', test(1:5,:),'Index',0)
% [Dstta,Qstta] = mspcAdicov(Lmodel,'Test',test(6:10,:),'Index',0)
%
%
% coded by: Jose Camacho (josecamacho@ugr.es)
% last modification: 22/Nov/2024
%
% Copyright (C) 2024  University of Granada, Granada
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

[ok, Lmodel] = checkLmodel(Lmodel);

N = Lmodel.nc;
M = size(Lmodel.XX, 2);
% Introduce optional inputs as parameters (name-value pair) 
p = inputParser;
addParameter(p,'Test',[]);
addParameter(p,'Index',1);
parse(p,varargin{:});

% Extract inputs from inputParser for code legibility
test = p.Results.Test;
index = p.Results.Index;

L = size(test, 1);

Lmodel.lvs = unique(Lmodel.lvs);
Lmodel.lvs(find(Lmodel.lvs==0))=[];
A = length(Lmodel.lvs);

% Validate dimensions of input data
if ~isempty(test), assert (isequal(size(test), [L M]), 'Dimension Error: 2nd argument must be L-by-M. Type ''help %s'' for more info.', routine(1).name); end
assert (isequal(size(index), [1 1]), 'Dimension Error: 3rd argument must be 1-by-1. Type ''help %s'' for more info.', routine(1).name); 

% Validate values of input data
assert (index==0 || index==1, 'Value Error: 3rd argument must be 0 or 1. Type ''help %s'' for more info.', routine(1).name);


%% Main code
    
if Lmodel.N        
    Lm = Lmodel;
    Lm.lvs = 1:size(Lm.XX,1); 
    if max(Lm.lvs) > 0
        if strcmp(Lmodel.type,'PLS')
            Lm = Lpls(Lm); 
            R = Lm.altweights;
            P = Lm.loads;
            d = Lm.sdT;
        elseif strcmp(Lmodel.type,'PCA')
            Lm = Lpca(Lm);
            P = Lm.loads; 
            R = P;
            d = Lm.sdT;
        end
        
        res = 1:size(R,2);
        indr=[];
        for i=1:length(Lmodel.lvs)
            indr = [indr find(res==Lmodel.lvs(i))];
        end
        res(indr) = [];
        
        onesV = ones(size(test,1),1);
        
        tests = preprocess2Dapp(test,Lmodel.av,'Scale',Lmodel.sc,'Weight',Lmodel.weight);
        
        if ~isempty(Lmodel.lvs)
            ti = adicov(Lmodel.XX,tests,length(Lmodel.lvs),'InSubspace',R(:,Lmodel.lvs),'OutSubspace',P(:,Lmodel.lvs));
        else
            ti = tests;
        end
        if ~isempty(res)
            ri = adicov(Lmodel.XX,tests,length(res),'InSubspace',R(:,res),'OutSubspace',P(:,res));
        else
            ri = tests;
        end
            
        if isempty(R(:,Lmodel.lvs))
            iti = 0;
        else
            iti = ADindex(tests,ti,'InSubspace',R(:,Lmodel.lvs)*diag(1./d(Lmodel.lvs)),'Index',index);
        end
        if isempty(R(:,res))
            iri = 0;
        else
            iri = ADindex(tests,ri,'InSubspace',R(:,res),'Index',index);
        end
        Dstt = iti;
        Qstt = iri;
        Rt = tests - ti;
        Rq = tests - ri;
    else
        Dstt = 0;
        Qstt = 0;
    end
else
    Dstt = 0;
    Qstt = 0;
end
 
