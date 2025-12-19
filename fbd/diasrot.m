function [R, P, T1u, Er, Ep] = diasrot(T1o, T2o, F1, F2)
%% nnspm (Next Nearest Permutation Matrix)
% computes the closest signed permutation matrix
% to a given rotation matrix R, allowing for reflections (i.e., entries of -1).
%
% [P, froError] = nnspm(R)
%
% Inputs:
% R - an n-by-n rotation matrix.
%
% Outputs:
%   P - the closest signed permutation matrix to R.
%   froError - the Frobenius norm of the difference, ||R - P||_F.
%
% The algorithm works by solving an assignment problem to maximize the sum
% of the absolute values of the selected entries in R. It uses MATLAB's
% matchpairs function (available in the Statistics and Machine Learning Toolbox)
% to find the optimal assignment.
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

[T1u, ord1, ~] = uniquetol(T1o,1e-6, 'ByRows', true, 'PreserveRange', true);
[T2u, ord2, ~] = uniquetol(T2o,1e-6, 'ByRows', true, 'PreserveRange', true);

lvls1 = F1(ord1, 1);
lvls2 = F2(ord2, 1);

%Orient levels according to T1u
[~, perm_idx] = ismember(lvls1, lvls2);
n = numel(lvls1);
P = eye(n);
P = P(perm_idx, :);
T2ua = P * T2u;

M = T1u' * T2ua;
[Up, ~, Vp] = svd(M);
R = Up * Vp';  % Rotation matrix
Er = T1u * R - T2ua;

n = size(R, 1);

% Form the cost matrix as the negative absolute value of R
costMat = -abs(R);

% Solve the assignment problem using matchpairs (Hungarian algorithm)
% The second argument (-Inf) ensures a complete assignment.
assignment = matchpairs(costMat, 1e6);

% Initialize the signed permutation matrix P
P = zeros(n);

% For each assignment, set the corresponding entry in P to the sign of R
for k = 1:size(assignment, 1)
    i = assignment(k, 1);
    j = assignment(k, 2);
    if R(i, j) >= 0
        P(i, j) = 1;
    else
        P(i, j) = -1;
    end
end

Ep = T1u*P - T2ua;
end