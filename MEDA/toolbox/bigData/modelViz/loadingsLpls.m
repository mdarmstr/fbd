
function [P,W,Q,figH] = loadingsLpls(Lmodel,varargin)

% Compute and plot loadings in PLS for large data.
%
% loadingsLpls(Lmodel) % minimum call
%
% INPUTS:
%
% Lmodel: (struct Lmodel) model with the information to compute the PLS
%   model:
%       Lmodel.XX: [MxM] X-block cross-product matrix.
%       Lmodel.XY: [MxO] cross-product matrix between the x-block and the
%           y-block.
%       Lmodel.lvs: [1x1] number of Latent Variables.
%       Lmodel.vclass: [Mx1] class associated to each variable.
%       Lmodel.varl: {ncx1} label of each variable.
%
% Optional INPUTS (parameters):
%
% 'Option': (str or num) options for data plotting: binary code of the form 'abc' for:
%       a:
%           0: no plots
%           1: plot loadings
%       b:
%           0: scatter plot of pairs of LVs
%           1: bar plot of each single LV
%       c:
%           0: plot weights
%           1: plot X-block loadings
%   By deafult, opt = '100'. If less than 3 digits are specified, least 
%   significant digits are set to 0, i.e. opt = 1 means a=1, b=0 and c=0. 
%   If a=0, then b and c are ignored.
%
%
% OUTPUTS:
%
% P: [MxA] X-block loadings
%
% W: [MxA] X-block weights
%
% Q: [OxA] Y-block loadings
%
% figH: (Lx1) figure handles
%
%
% EXAMPLE OF USE: Random loadings: bar and scatter plot of loadings
%
% X = simuleMV(20,10,'LevelCorr',8);
% Y = 0.1*randn(20,2) + X(:,1:2);
% Lmodel = iniLmodel(X,Y);
% Lmodel.lvs = 1;
% loadingsLpls(Lmodel);
% Lmodel.lvs = 1:2;
% loadingsLpls(Lmodel);
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
    
%% Arguments checking

% Set default values
routine=dbstack;
assert (nargin >= 1, 'Error in the number of arguments. Type ''help %s'' for more info.', routine(1).name);
[ok, Lmodel] = checkLmodel(Lmodel);

% Introduce optional inputs as parameters (name-value pair) 
p = inputParser;
addParameter(p,'Option','100');
parse(p,varargin{:});

% Extract inputs from inputParser for code legibility
opt = p.Results.Option;

% Convert int arrays to str
if isnumeric(opt), opt=num2str(opt); end

% Convert int arrays to str
while length(opt)<3, opt = strcat(opt,'0'); end

% Validate dimensions of input data
assert (~isempty(Lmodel.XY), 'Dimension Error: Empty XY. Type ''help %s'' for more info.', routine(1).name);
assert (ischar(opt) && length(opt)==3, 'Dimension Error: 2nd argument must be a string or num of maximum 3 bits. Type ''help %s'' for more info.', routine(1).name);
  
% Validate values of input data
assert (isempty(find(opt~='0' & opt~='1')), 'Value Error: 2nd argument must contain binary values. Type ''help %s'' for more info.', routine(1).name);


%% Main code

Lmodel = Lpls(Lmodel);
W = Lmodel.weights;
Q = Lmodel.yloads;
P = Lmodel.loads;


%% Show results

figH = [];
if opt(1) == '1'
    
    if opt(3) == '0'
        Pt = W;
        text = 'Weights';
    else
        Pt = P;
        text = 'X-block loadings';
    end
    
    [yvar,tvar] = varLpls(Lmodel,'Plot',false);
     
    if length(Lmodel.lvs) == 1 || opt(2) == '1'
        for i=1:length(Lmodel.lvs)
            figH(i) = plotVec(Pt(:,i),'EleLabel',Lmodel.varl,'ObsClass',Lmodel.vclass,'XYLabel',{'',sprintf('%s LV %d (%.0f%%)',text,Lmodel.lvs(i),100*(tvar(i) - tvar(i+1)))});
        end
    else
        h = 1;
        for i=1:length(Lmodel.lvs)-1
            for j=i+1:length(Lmodel.lvs)
                figH(h) = plotScatter([Pt(:,i),Pt(:,j)],'EleLabel',Lmodel.varl,'ObsClass',Lmodel.vclass,'XYLabel',{sprintf('%s LV %d (%.0f%%)',text,Lmodel.lvs(i),100*(tvar(i) - tvar(i+1))),sprintf('%s LV %d (%.0f%%)',text,Lmodel.lvs(j),100*(tvar(j) - tvar(j+1)))}');
                h = h+1;
            end      
        end
    end
end
        