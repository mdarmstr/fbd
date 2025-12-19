function [p,T1oe,T1r,T2oe,R] = nnspt(parglmoA, parglmoB, n_perms)

%% Next Nearest Signed Permutation Test
% Tests whether the scores in parglmoA are independently distributed with 
% respect to paglmoB up to sign and permutational ambiguities. Utilizes a
% two-sided test via the chi^2_1 distribution.
%
% **IMPORTANT NOTE: Test all factors for evidence of significance across
% parglmoA and parglmoB. Reinitialize with only the factors of interest in
% F1, F2 **
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
% Install Statistics and Machine Learning Toolbox for MATLAB
%
% coded by: Michael Sorochan Armstrong (mdarmstr@ugr.es)
%           Jose Camacho Paez (josecamacho@ugr.es)
%
% last modification: DEC 2025
%
% Copyright (C) 2025  University of Granada, Granada
% Copyright (C) 2025  Michael Sorochan Armstrong, Jose Camacho Paez
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

% Extract data
X1 = parglmoA.data;
X2 = parglmoB.data;

F1 = parglmoA.design;
F2 = parglmoB.design;

% Ensure design matrices are sorted in the correct order
[F1,idx] = sort(F1,"ascend");
X1 = X1(idx,:);
Z1 = parglmoA.D(idx,:);

[F2,idx] = sort(F2,"ascend");
X2 = X2(idx,:);
Z2 = parglmoB.D(idx,:);

% Predicted coefficients
B1hat = pinv(Z1)*X1;
X1n = Z1*B1hat;
E1 = X1 - X1n;

B2hat = pinv(Z2)*X2;
X2n = Z2*B2hat;
E2 = X2 - X2n; 

% Nominal score calculation
[~, ~, V1] = svds(X1n, rank(X1n));
[~, ~, V2] = svds(X2n, rank(X2n));

T1o = X1n * V1 ;
T2o = X2n * V2 ;

% Calculate diasmetic statistic
[R,P,T1u,Er,Ep] = diasrot(T1o,T2o,F1,F2);

% Incorporate noise and apply the rotation
T1oe = ((X1n + E1) * V1);    % T1 with noise (before rotation)
T1r = T1oe * R;              % T1 after rotation
T2oe = (X2n + E2) * V2;      % T2 after noise (no rotation)

N = size(T1u,1);

Sobs  = (Er'*Er + Ep'*Ep) / (N);
Siobs = pinv(Sobs);
Lobs  = chol(Siobs,'lower');

Fd = norm(T1u*(P - R)*Lobs,'fro')^2;
Fp = zeros([1,n_perms]);

for ii = 1:n_perms
    % TIII permutation
    perms = randperm(size(E1,1));
    Eperm = E1(perms,:);
    Xperm = X1n + Eperm;

    % Recalculate GLM
    pD1 =  pinv(Z1);
    Bperm = pD1*Xperm;
    X1perm = Z1*Bperm;
    
    % Calculate unique, permuted scores
    Tpm = X1perm * V1;
    [Tpu, ~, ~] = uniquetol(Tpm,1e-6, 'ByRows', true, 'PreserveRange', true);

    % Permutational test statistic
    Fp(ii) = norm(Tpu*(P - R)*Lobs,'fro')^2;

end

% chi2 test statistic
chid = (Fd - mean(Fp)) / std(Fp);
chid = chid^2;

% Permutational chi2 test distribution
chip = (Fp - mean(Fp)) / std(Fp);
chip = chip.^2; 

% Probability of rejecting the null hypothesis
p = (sum(chid >= chip) + 1) / (n_perms + 1);

end