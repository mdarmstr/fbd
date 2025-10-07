function ascao = asca(parglmo)

% ASCA is a data analysis algorithm for designed experiments. The input is 
% a General Linear Models (GLM) factorization of the data (done with parglm 
% and stored in parglmo) and the code applies Principal Component Analysis 
% to the factor/interaction matrices.
%
% ascao = asca(parglmo)   % minimum call
%
%
% See also: parglm, apca, gasca, vasca, createDesign
%
%
% INPUTS:
%
% parglmo (structure): structure with the GLM decomposition with factor and 
% interaction matrices, p-values and explained variance. 
%
%
% OUTPUTS:
%
% ascao (structure): structure that contains scores, loadings, singular
% values and projections of the factors and interactions.
%
%
% EXAMPLE OF USE (copy and paste the code in the command line)
%   Random data, two factors, with 4 and 3 levels, but only the first one 
%   is significative, and 4 replicates:
%
% reps = 4;
% vars = 400;
% levels = {[1,2,3,4],[1,2,3]};
% 
% F = createDesign(levels,'Replicates',reps);
% 
% X = zeros(size(F,1),vars);
% for i = 1:length(levels{1}),
%     X(find(F(:,1) == levels{1}(i)),:) = simuleMV(length(find(F(:,1) == levels{1}(i))),vars,'LevelCorr',8) + repmat(randn(1,vars),length(find(F(:,1) == levels{1}(i))),1);
% end
% 
% [table, parglmo] = parglm(X, F);
% table
% 
% ascao = asca(parglmo);
% 
% for i=1:2, % Note, the second factor is only shown for the sake of illustration, but non-significant factors should not be visualized
%   scores(ascao.factors{i},'Title',sprintf('Factor %d',i),'ObsClass',ascao.design(:,i));
%   loadings(ascao.factors{i},'Title',sprintf('Factor %d',i));
% end
%
%
% EXAMPLE OF USE (copy and paste the code in the command line)
%   Random data, two significative factors, with 4 and 3 levels, and 4 replicates:
%
% reps = 4;
% vars = 400;
% levels = {[1,2,3,4],[1,2,3]};
% 
% F = createDesign(levels,'Replicates',reps);
% 
% X = zeros(size(F,1),vars);
% for i = 1:length(levels{1}),
%     fi{i} = randn(1,vars);
% end
% for j = 1:length(levels{2}),
%     fj{j} = randn(1,vars);
% end
% for i = 1:length(levels{1}),
%     for j = 1:length(levels{2}),
%         X(find(F(:,1) == levels{1}(i) & F(:,2) == levels{2}(j)),:) = simuleMV(reps,vars,'LevelCorr',8) + repmat(fi{i} + fj{j},reps,1);
%     end
% end
% 
% [table, parglmo] = parglm(X, F, 'Model',{[1 2]});
% table
% 
% ascao = asca(parglmo);
% 
% for i=1:2,
%   scores(ascao.factors{i},'Title',sprintf('Factor %d',i),'ObsClass',ascao.design(:,i));
%   loadings(ascao.factors{i},'Title',sprintf('Factor %d',i));
% end
%
%
% EXAMPLE OF USE (copy and paste the code in the command line)
%   Random data, two factors with 4 and 3 levels, and 4 replicates, with 
%   significant interaction:
%
% reps = 4;
% vars = 400;
% levels = {[1,2,3,4],[1,2,3]};
% 
% F = createDesign(levels,'Replicates',reps);
% 
% X = zeros(size(F,1),vars);
% for i = 1:length(levels{1}),
%     for j = 1:length(levels{2}),
%         X(find(F(:,1) == levels{1}(i) & F(:,2) == levels{2}(j)),:) = simuleMV(reps,vars,'LevelCorr',8) + repmat(randn(1,vars),reps,1);
%     end
% end
% 
% [table, parglmo] = parglm(X, F, 'Model',{[1 2]});
% table
% 
% ascao = asca(parglmo);
% 
% codeLevels = {};
% for i=1:size(F,1), codeLevels{i} = sprintf('F1:%d,F2:%d',F(i,1),F(i,2));end;
% ascao.interactions{1}.lvs = 1:2;
% scores(ascao.interactions{1},'Title','Interaction','ObsClass',codeLevels);
% loadings(ascao.interactions{1},'Title','Interaction');
%
%
% Coded by: Jose Camacho (josecamacho@ugr.es)
% Last modification: 4/Apr/2025
% Dependencies: Matlab R2017b, MEDA v1.8
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


%% Main code

ascao = parglmo;

%Do PCA on level averages for each factor
for factor = 1 : ascao.nFactors
    
    xf = ascao.factors{factor}.matrix;
    model = pcaEig(xf,'PCs',1:rank(xf));
    
    fnames = fieldnames(model);
    for n = 1:length(fnames)
        ascao.factors{factor} = setfield(ascao.factors{factor},fnames{n},getfield(model,fnames{n}));
    end
    
    if isempty([ascao.factors{factor}.refF ascao.factors{factor}.refI])
        ascao.factors{factor}.scoresV = (xf+ascao.residuals)*model.loads;
    else
        ascao.factors{factor}.scoresV = xf;
        for n = 1:length(ascao.factors{factor}.refF) 
            ascao.factors{factor}.scoresV = ascao.factors{factor}.scoresV + ascao.factors{ascao.factors{factor}.refF(n)}.matrix;
        end
        for n = 1:length(ascao.factors{factor}.refI) 
            ascao.factors{factor}.scoresV = ascao.factors{factor}.scoresV + ascao.interactions{ascao.factors{factor}.refI(n)}.matrix;
        end
        ascao.factors{factor}.scoresV = ascao.factors{factor}.scoresV*model.loads;
    end
end

%Do PCA on interactions
for interaction = 1 : ascao.nInteractions
    
    xf = ascao.interactions{interaction}.matrix;
    for factor = ascao.interactions{interaction}.factors
        xf = xf + ascao.factors{factor}.matrix;
    end
    model = pcaEig(xf,'PCs',1:rank(xf));
    
    fnames = fieldnames(model);
    for n = 1:length(fnames)
        ascao.interactions{interaction} = setfield(ascao.interactions{interaction},fnames{n},getfield(model,fnames{n}));
    end

    if isempty(ascao.interactions{interaction}.refI)
        ascao.interactions{interaction}.scoresV = (xf+ascao.residuals)*model.loads;
    else
        ascao.interactions{interaction}.scoresV = xf;
        for n = 1:length(ascao.interactions{interaction}.refI) 
            ascao.interactions{interaction}.scoresV = ascao.interactions{interaction}.scoresV + ascao.interactions{ascao.interactions{interaction}.refI(n)}.matrix;
        end
        ascao.interactions{interaction}.scoresV = ascao.interactions{interaction}.scoresV*model.loads;
    end
end

ascao.type = 'ASCA';

