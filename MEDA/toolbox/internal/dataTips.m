function output_txt = dataTips(~, event_obj, bdata, varargin)
% Data tips for scatter and vec plots
%
% See also: plotScatter, plotVec
%
% INPUTS:
%
% bdata: Data used in the plots.
%   (Nx2) for scatter plots
%   (Nx1) for vec plots
%
% Optional INPUTS (parameters):
%
% 'EleLabel': [Nx1] name of the elements
%
% 'ObsClass': [Nx1] groups for different visualization
%
% 'ClassType': str. Class type of the data (Categorial or numerical). Default = "Categorical".
%
% 'Multiplicity': [NxM] multiplicity of each row (1s by default)
%
% 'ObsAlpha': [Nx1] opacity values for every element (1s by default)
%
% OUTPUTS:
%
% output_txt: dataTip for point/bar.
%
% coded by: Jesús García Sánchez (gsus@ugr.es)
%           
% last modification: 01/Jun/2025
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

%

%% Parameters checking

% Set default values
routine=dbstack;
assert (nargin >= 2, 'Error in the number of arguments. Type ''help %s'' for more info.', routine(1).name);

N = size(bdata, 1);

% Introduce optional inputs as parameters (name-value pair) 
p = inputParser;
addParameter(p,'EleLabel',[]);   
addParameter(p,'Classes',[]);
addParameter(p,'ClassType',"Categorical");
addParameter(p,'Multiplicity',ones(N,1));
addParameter(p,'ObsAlpha',ones(N,1))
parse(p,varargin{:});

% Extract inputs from inputParser for code legibility
elabel = p.Results.EleLabel;
classes = p.Results.Classes;
classType = p.Results.ClassType;
mult = p.Results.Multiplicity;
alphas = p.Results.ObsAlpha;


if iscell(classes)
    classes = string(classes);
elseif isnumeric(classes)
    classes = string(classes);
elseif iscategorical(classes)
    classes = string(classes);
end

if iscell(elabel)
    elabel = string(elabel);
elseif isnumeric(elabel)
    elabel = string(elabel);
elseif iscategorical(elabel)
    elabel = string(elabel);
end

%% Main code

    pos = get(event_obj, 'Position');
    target = get(event_obj, 'Target');
    obj_type = class(target);

    if obj_type == "matlab.graphics.chart.primitive.Bar"  % Bar plot
        idx = round(pos(1));
    elseif size(bdata,2) == 2 % Scatter
        idx = find(bdata(:,1) == pos(1) & bdata(:,2) == pos(2));
    else
        idx = pos(1);
    end
    output_txt = {
        ['x: ', num2str(pos(1))], ...
        ['y: ', num2str(pos(2))],    };

    
    if ~isempty(elabel) || ~isempty(classes) % Newline
        output_txt{end+1} = ''; end

    if ~isempty(elabel)
        output_txt{end+1} = ['Observation: ', num2str(elabel(idx))];
    end

    if ~isempty(classes)
        if classType == "Categorical"
        output_txt{end+1} = ['Class: ', num2str(classes(idx))]; end
        if classType == "Numerical"
        output_txt{end+1} = ['Value: ', num2str(classes(idx))]; end
    end

    if any(mult ~= 1)
        output_txt{end+1} = ['Multiplicity: ', num2str(mult(idx))];
    end

    if any(alphas ~= 1)
        output_txt{end+1} = ['Opacity: ', num2str(alphas(idx))];
    end

end