function [bpvals, pboot] = pbootasca(X,F,ascao,nfact,varargin)

% Bootstraping in ASCA models.
%
% bpvals = pbootasca(X,F,ascao,nfact) % minimum call
%
% See also: asca, parglm, parglmVS, parglmMC, createDesign
%
%
% INPUTS:
%
% X: [NxM] billinear data set for model fitting, where each row is a
% measurement, each column a variable
%
% F: [NxF] design matrix, where columns correspond to factors and rows to
% levels.
%
% ascao: (structure) structure that contains scores, loadings, singular
% values and projections of the factors and interactions in ASCA.
%
% nfact: [1x1] factor where bootstrapping is performed
%
%
% Optional INPUTS (parameters):
%
% 'NRuns': [1x1] number of runs (1000 by default)
%
% 'Plot': (bool) plot results
%       false: no plots.
%       true: plot (default)
%
% 'PlotVal': [1x1] singificance for the intervals to plot (0.01 by default)
%
%
% OUTPUTS:
%
% bpvals: p-value per variables
%
% pboot: bootstrapped loadings
%
%
% EXAMPLE OF USE: Random data, one factor and two levels, three variables 
% with information on the factor. This example takes long to compute, you  
% may reduce the number of variables or permutations.
%
% nObs = 40;
% nVars = 100;
% 
% class = (randn(nObs,1)>0)+1;
% X = simuleMV(nObs,nVars,'LevelCorr',8);
% X(class==2,1:3) = X(class==2,1:3) + 10;
% 
% S = rng; % Use same seed for random generators to improve comparability of results
% [T, parglmo] = parglm(X,class); % No variables selection 
% rng(S);
% [TVS, parglmoVS] = parglmVS(X, class); % With multivariate variable selection
% 
% ascao = asca(parglmo); % With variable selection through bootstrapping
% bpvals = pbootasca(X, class, ascao, 1, 'NRuns',1000);
% 
% h = figure; hold on
% plot([1 nVars],[parglmo.p parglmo.p],'b-.')
% plot(parglmoVS.p(parglmoVS.ordFactors),'g-o')
% plot(bpvals(parglmoVS.ordFactors),'k-')
% plot([0,size(X,2)],[0.05 0.05],'r:')
% plot([0,size(X,2)],[0.01 0.01],'r--')
% legend('ASCA','VASCA','Bootstrap','alpha=0.05','alpha=0.01','Location','southeast')
% a=get(h,'CurrentAxes');
% set(a,'FontSize',14)
% ylabel('p-values','FontSize',18)
% xlabel('Variables in selected order','FontSize',18)
%
%
% coded by: Rafa Vitale (raffaele.vitale@univ-lille.fr)
%           Jose Camacho (josecamacho@ugr.es)
% last modification: 16/Jan/20245
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
assert (nargin >= 4, 'Error in the number of arguments. Type ''help %s'' for more info.', routine(1).name);
N = size(X, 1);
M = size(X, 2);


% Introduce optional inputs as parameters (name-value pair) 
p = inputParser;
addParameter(p,'NRuns',1000); 
addParameter(p,'Plot',true);
addParameter(p,'PlotVal',0.1);  
parse(p,varargin{:});

% Extract inputs from inputParser for code legibility
nboot = p.Results.NRuns;
opt = p.Results.Plot;
pvalue = p.Results.PlotVal;

% Validate dimensions of input data
assert (isequal(size(nfact), [1 1]), 'Dimension Error: parameter ''nfact'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(nboot), [1 1]), 'Dimension Error: parameter ''NRuns'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(pvalue), [1 1]), 'Dimension Error: parameter ''PlotVal'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);


%% Main code

lev = F(:,nfact);
p = ascao.factors{nfact}.loads;
pboot = zeros([nboot,size(p)]); 

model = [];
for i=1:length(ascao.interactions)
    model = [model; ascao.interactions{i}.factors];
end

for boot=1:nboot
    
    uF = unique(lev);
    for n1 = 1:length(uF)
        
        ind=find(lev==uF(n1));
        bootind=ceil(length(ind)*rand(length(ind),1));
        yboot(ind,:)=X(ind(bootind),:);
        
    end
    
    [~, parglmo] = parglm(yboot,F,'Model',model,'Preprocessing',ascao.prep,'Permutations',ascao.nPerm,'Ts',ascao.ts,'Ordinal',ascao.ordinal,'Fmtc',ascao.fmtc,'Coding',ascao.coding,'Nested',ascao.nested); 
    ascao = asca(parglmo); 
    pb = ascao.factors{nfact}.loads; 
    [~,pboot(boot,:,:)] = orthProc(p,pb);
    
end

for i=1:size(pboot,3)
    for j=1:size(pboot,2)
        bpvals(j,i) = (min(length(find(pboot(:,j,i)<0)),length(find(pboot(:,j,i)>=0)))+1)/(nboot+1);
    end
end
bpvals = min(bpvals,[],2)';

%% Plot

if opt, evalboot(p,pboot,pvalue); end


% Orthogonal procrustes
function [r,yrot]=orthProc(x,y)

%Computes orthogonal procrustes rotation projecting matrix y onto the
%subspace spanned by matrix x.
% syntax: [r,yrot]=orthProc(x,y)
% where r is the rotation matrix and yrot is the procrustes rotated y

[u,~,v]=svd(y'*x,0);
r=u*v';
yrot=y*r;


% Viz
function evalboot(p,pboot,pvalue)

% Plots results

for npc=1:size(pboot,3)
    
    plotVec(p(:,npc), 'XYLabel', {'',sprintf('Loadings PC %d',npc)});
    hold on
    
    for nvar=1:size(p,1)
        
        PM = prctile(squeeze(pboot(:,nvar,npc)),100);
        Pm = prctile(squeeze(pboot(:,nvar,npc)),0);
        if abs(PM)>abs(Pm)
            Pm = prctile(squeeze(pboot(:,nvar,npc)),pvalue*100);
        else
            PM = prctile(squeeze(pboot(:,nvar,npc)),(1-pvalue)*100);
        end
        
        line([nvar nvar],[Pm PM],'LineWidth',2,'Color','k')
        line([nvar-.25 nvar+.25],[PM PM],'LineWidth',2,'Color','k')
        line([nvar-.25 nvar+.25],[Pm Pm],'LineWidth',2,'Color','k')
        
    end
        
     axis tight
     axes=axis;
     axis([0 size(p,1)+1 axes(3) axes(4)])
    
end