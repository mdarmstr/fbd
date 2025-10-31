function [p,T1oe,T1r,T2oe,R] = nnspt(parglmoA, parglmoB, F1, F2, fctrs, n_perms)

%% Fusion by Design (FBD)
%
% INPUTS
% parglmoA - output of parglm.m in MEDA toolbox for X1, F1
% parglmoB - output of parglm.m in MEDA toolbox for X2, F2
% F1       - Experimental factor matrix for X1
% F2       - Experimental factor matrix for X2
% fctrs    - List of factors {1,2,[1,2]} for analysis by FBD
% n_perms  - Number of permutations required for null hypothesis testing
% OUTPUTS
% p        - p-value for NNSPM heteromodal statistical test
% T1oe     - Original PCA scores for X1 + E1, with respect to fctrs
% T1r      - Rotated PCA scores for X1 + E1, with respect to fctrs, X2, F2
% T2oe     - Original PCA scores for X2 + E2, with respect to fctrs
%
% Software preparation:  Install MEDA-Toolbox following readme file;
%
% coded by: Michael Sorochan Armstrong (mdarmstr@ugr.es)
%           Jose Camacho Paez (josecamacho@ugr.es)
%
% last modification: /Oct/2025
%
% Copyright (C) 2025  University of Granada, Granada
% Copyright (C) 2025  Jose Camacho Paez, Michael Sorochan Armstrong
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
% MEDA dependencies: parglmo

X1 = parglmoA.data;
%X2 = parglmoB.data;
 
X1ne = parglmoA.residuals;
X2ne = parglmoB.residuals;

% identify interacting terms, linear terms
% note the order must be [1,2], and not [2,1]. See parglm for details
intr_logical = cellfun(@(x) numel(x)>1, fctrs);
iidx = zeros(1,sum(intr_logical));
intr = fctrs(intr_logical);
iidx_count = 1;

if ~isempty(intr_logical) && any(intr_logical)
    for ii = 1:sum(intr_logical)
        lint = parglmoA.interactions{ii}.factors;
        if isequal(intr{ii},lint)
            iidx(ii) = iidx_count;
        end
        iidx_count = iidx_count + 1;
    end
end

fctr_logical = cellfun(@(x) isscalar(x), fctrs);
fidx = zeros(1,sum(fctr_logical));
fctr = fctrs(fctr_logical);
fidx_count = 1;

if ~isempty(fctr_logical) && any(fctr_logical)
    for ii = 1:sum(fctr_logical)
        fint = parglmoA.factors{ii}.factors;
        if isequal(fctr{ii},fint)
            fidx(ii) = fidx_count;
        end
        fidx_count = fidx_count + 1;
    end
end

D1  = [];
X1n = zeros(size(parglmoA.data,1),size(parglmoA.data,2));
X2n = zeros(size(parglmoB.data,1),size(parglmoB.data,2));

for ii = 1:length(iidx)
    X1n = X1n + parglmoA.interactions{iidx(ii)}.matrix; 
    X2n = X2n + parglmoB.interactions{iidx(ii)}.matrix;
    D1 = [D1,parglmoA.D(:,parglmoA.interactions{iidx(ii)}.DVars)]; %#ok<AGROW>
end

for ii = 1:length(fidx)
    X1n = X1n + parglmoA.factors{fidx(ii)}.matrix;
    X2n = X2n + parglmoB.factors{fidx(ii)}.matrix;
    D1 = [D1,parglmoA.D(:,parglmoA.factors{fidx(ii)}.Dvars)]; %#ok<AGROW>
end

%X1n = parglmoA.factors{fctrs(1)}.matrix;
%X2n = parglmoB.factors{fctrs(1)}.matrix;

%D1 = parglmoA.D(:,parglmoA.factors{fctrs(1)}.Dvars);
%D2 = parglmoB.D(:,parglmoB.factors{fctrs(1)}.Dvars);

% Scores calculation

[~, ~, V1] = svds(X1n, rank(X1n));
[~, ~, V2] = svds(X2n, rank(X2n));

% VARIMAX rotation
[V1,T] = rotatefactors(V1,'Method','varimax','maxit',5000,'reltol',1e-12);
V2 = rotatefactors(V2,'Method','varimax','maxit',5000,'reltol',1e-12);

T1o = X1n * V1;
T2o = X2n * V2;

% Procrustes rotation as a separate script
R = procrustes_rot(T1o,T2o,F1,F2);

% Incorporate noise and apply the rotation
T1oe = ((X1n + X1ne) * V1);    % T1 with noise (before rotation)
T1r = T1oe * R;                % T1 after rotation
T2oe = (X2n + X2ne) * V2;      % T2 after noise (no rotation)

%Calculate next nearest signed permutation statistic
[~,Fd] = nnspm(R);

%Permutation test
Fp = zeros([1,n_perms]);

for ii = 1:n_perms
    perms = randperm(size(X1,1)); % permuted data (permute whole data matrix)
    Xperm = X1(perms, :);
    pD1 =  pinv(D1'*D1)*D1';
    Bperm = pD1*Xperm;
    X1perm = D1*Bperm;
    [~,~,Vpm] = svds(X1perm,rank(X1perm));
    Tpm = X1perm * Vpm * T;
    R = procrustes_rot(Tpm,T2o,F1,F2);
    [~, err] = nnspm(R);
    Fp(ii) = err;
end

p = (sum(Fp < Fd) + 1) / (n_perms + 1);

end