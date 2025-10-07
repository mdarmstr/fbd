function xvar = varLpca(Lmodel,varargin)

% Variability captured in terms of the number of PCs.
%
% varLpca(Lmodel) % minimum call
% xvar = varLpca(Lmodel,'Option',1) %complete call
%
%
% INPUTS:
%
% Lmodel: (struct Lmodel) model with the information to compute the PCA
%   model:
%       Lmodel.XX: (MxM) X-block cross-product matrix.
%       Lmodel.lvs: (1x1) number of PCs.
%
% Optional INPUTS (parameters):
%
% 'Plot': (bool) plot results
%       false: no plots.
%       true: plot (default)
%
%
% OUTPUTS:
%
% xvar: [Ax1] Percentage of captured variance of X.
%
%
% EXAMPLE OF USE: Random data
%
% Lmodel = iniLmodel(simuleMV(20,10,'LevelCorr',8));
% Lmodel.lvs = 0:10;
% xvar = varLpca(Lmodel);
%
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
assert (nargin >= 1, 'Error in the number of arguments. Type ''help %s'' for more info.', routine(1).name);

% Introduce optional inputs as parameters (name-value pair) 
p = inputParser;
addParameter(p,'Plot',true);
parse(p,varargin{:});

% Extract inputs from inputParser for code legibility
opt = p.Results.Plot;

checkLmodel(Lmodel);

% Preprocessing
Lmodel.lvs = unique([0 Lmodel.lvs]);


%% Main code

Lmodel.lvs = 0:max(Lmodel.lvs);
Lmodel = Lpca(Lmodel);
P = Lmodel.loads;

totalVx = sum(eig(Lmodel.XX));
xvar = ones(max(Lmodel.lvs)+1,1);
for i=1:max(Lmodel.lvs)
    xvar(i+1) = xvar(i+1) - sum(eig(P(:,1:i)'*Lmodel.XX*P(:,1:i)))/totalVx;
end
    
%% Show results

if opt 
    plotVec(xvar,'EleLabel',[0 Lmodel.lvs],'XYLabel',{'#PCs','% Residual Variance'},'PlotType','Lines');
end
