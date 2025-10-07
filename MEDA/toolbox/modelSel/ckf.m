function [cumpress,press,term1,term2,term3] = ckf(xcs,T,P,varargin)
%
% CKF Algorithm: Journal of Chemometrics, 29(8): 467-478, 2015
%
% cumpress = ckf(xcs,T,P) % minimum call
%
%
% INPUTS:
%
% xcs: [NxM] preprocessed billinear data set 
%
% T: [NxA] scores.
%
% P: [MxA] loadings.
%
%
% Optional INPUTS (Parameter):
%
% 'Plot': (bool) plot results
%       false: no plots.
%       true: plot (default)
%
%
% OUTPUTS:
%
% cumpress: [Ax1] ckf curve.
%
% press: [AxM] PRESS per variable.
%
% term1: [NxM] first error term, according to referred paper.
%
% term2: [NxM] second error term, according to referred paper.
%
% term3: [NxM] third error term, according to referred paper.
%
%
% EXAMPLE OF USE: Random curve, two examples of use.
%
% X = simuleMV(20,10,'LevelCorr',8);
% model = pcaEig(X);
% P = model.loads;
% T = model.scores;
% 
% % To Plot ('Option' default 1)
% cumpress = ckf(X,T,P);
% 
% % Not to plot
% cumpress = ckf(X,T,P,'Plot',0);
%
% coded by: Jose Camacho (josecamacho@ugr.es)
% last modification: 16/Jan/2025
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
assert (nargin >= 3, 'Error in the number of arguments. Type ''help %s'' for more info.', routine(1).name);
N = size(xcs, 1);
M = size(xcs, 2);
A = size(T, 2);


% Introduce optional inputs as parameters (name-value pair) 
p = inputParser;
addParameter(p,'Plot',true);             
parse(p,varargin{:});

% Extract inputs from inputParser for code legibility
opt = p.Results.Plot;

% Validate dimensions of input data
assert (isequal(size(T), [N A]), 'Dimension Error: parameter ''T'' must be N-by-A. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(P), [M A]), 'Dimension Error: parameter ''P'' must be M-by-A. Type ''help %s'' for more info.', routine(1).name);


%% Main code

cumpress = zeros(A+1,1);
press = zeros(A+1,size(P,1));

s = size(xcs);

for i=0:A
    
    if i > 0 % PCA Modelling
        p2 = P(:,1:i);
        srec = T(:,1:i)*p2';
        erec = xcs - srec;
        term3p = erec;
        term1p = xcs.*(ones(s(1),1)*(sum(p2.*p2,2))');
    else % Modelling with the average
        term1p = zeros(size(xcs));
        term3p = xcs;
    end
    
    term1 = term1p.^2;
    term2 = 2*term1p.*term3p;
    term3 = term3p.^2;
    
    press(i+1,:) = sum([sum(term1,1);sum(term2,1);sum(term3,1)]);
    
    cumpress(i+1) = sum(press(i+1,:));
end
    
    
%% Show results

if opt
    A = size(T, 2);
    Z = 0:A;
    figh = plotVec(cumpress/cumpress(1),'EleLabel',Z,'XYLabel',{'#PCs','ckf'},'PlotType','Lines'); 
end

        